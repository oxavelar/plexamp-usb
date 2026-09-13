#!/usr/init/env python3
"""Export Plex music to a car-friendly USB filesystem.

Discovers a reachable Plex Media Server, authenticates when unauthenticated
access is unavailable, selects music playlists, and exports tracks directly
into a local Downloads directory.

Supported source formats are copied directly when matching configured formats.
All other formats are converted using FFmpeg to the top-priority target
format (defaulting to AAC-LC VBR highest quality).

Operations are deterministic and resumable. Complete files are preserved, while
stale, incomplete, or invalid files are cleaned up or retried automatically.
Playlists and random-fill folders prune orphaned files during refreshes while
retaining pending or active targets.

Configuration is maintained in settings.json alongside this script.

Requirements:
    - Python 3.10+
    - ffmpeg and ffprobe for audio conversion
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import functools
import hashlib
import json
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator


APP_NAME = "plexamp-usb"
USER_AGENT = f"{APP_NAME}/1.0"
CONFIG_JSON = "settings.json"
CONFIG_PATH = Path(__file__).resolve().parent / CONFIG_JSON
DOWNLOAD_DIR = "Downloads"

DURATION_TOLERANCE_SECONDS = 2.0
ADAPTIVE_SUCCESS_THRESHOLD = 8
ADAPTIVE_MIN_WORKERS = 1
DOWNLOAD_CHUNK_BYTES = 64 * 1024
PART_SUFFIX = ".part"
AUDIO_EXTENSIONS = frozenset({".mp3", ".m4a", ".aac", ".flac", ".ogg"})

# Every container/codec/extension spelling Plex reports, mapped to one canonical format.
FORMAT_ALIASES = {
    "mp3": "mp3", "mpeg": "mp3",
    "aac": "aac", "m4a": "aac", "mp4": "aac", "mp4a": "aac",
    "flac": "flac",
    "ogg": "ogg", "oga": "ogg", "vorbis": "ogg", "opus": "ogg",
}
KNOWN_FORMATS = frozenset(FORMAT_ALIASES.values())
FORMAT_EXTENSIONS = {"aac": ".m4a", "mp3": ".mp3", "flac": ".flac", "ogg": ".ogg"}

RESERVED_NAMES = (
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)
ILLEGAL_CHARS_PATTERN = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
WHITESPACE_PATTERN = re.compile(r"\s+")
BITRATE_PATTERN = re.compile(r"[0-9]+K")
FORMAT_NAME_PATTERN = re.compile(r"[a-z0-9._+-]+")
SIZE_UNITS = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
RESERVE_PATTERN = re.compile(r"^([0-9]*\.?[0-9]+)\s*(%|[KMGT]B?|B)?$", re.IGNORECASE)

DEFAULT_CONFIG = {
    "plex": {
        "host": "",
        "port": 32400,
        "token": "",
        "user": "",
        "timeout": 30,
    },
    "audio": {
        "conversion_formats": ["aac:vbr", "mp3:vbr"],
        "conversion_threads": "auto",
    },
    "output": {
        "directory": DOWNLOAD_DIR,
        "directory_limit": 255,
        "reserve": "5%",  # Can be "5%", "128M", "2G", "500K", or raw bytes
    },
    "random": {
        "max_tracks": 1000,
        "strategy": "freshness",  # "freshness" or "random"
    },
    "download": {
        "retries": 3,
        "retry_delay": 2.0,
    },
}

ACTIVE_PROCESSES: set[subprocess.Popen] = set()
# In-flight .part files -> bytes written, or -1 when only the filesystem knows (ffmpeg).
PART_PROGRESS: dict[Path, int] = {}
# Reentrant so a Ctrl-C arriving while the main thread already holds it cannot deadlock.
STATE_LOCK = threading.RLock()


def part_path(destination: Path) -> Path:
    return destination.with_suffix(destination.suffix + PART_SUFFIX)


def _register_part_file(path: Path, written: int = -1) -> None:
    with STATE_LOCK:
        PART_PROGRESS[path] = written


def _update_part_progress(path: Path, written: int) -> None:
    # Only the owning worker touches its own key, and dict stores are atomic, so no lock.
    PART_PROGRESS[path] = written


def _unregister_part_file(path: Path) -> None:
    with STATE_LOCK:
        PART_PROGRESS.pop(path, None)


def active_part_bytes() -> int:
    """Bytes buffered in in-flight .part files, polling only the ones that cannot self-report."""
    with STATE_LOCK:
        snapshot = list(PART_PROGRESS.items())

    total = 0
    for path, written in snapshot:
        if written >= 0:
            total += written
            continue
        try:
            total += path.stat().st_size
        except OSError:
            pass
    return total


def active_part_count() -> int:
    with STATE_LOCK:
        return len(PART_PROGRESS)


def _register_process(proc: subprocess.Popen) -> None:
    with STATE_LOCK:
        ACTIVE_PROCESSES.add(proc)


def _unregister_process(proc: subprocess.Popen) -> None:
    with STATE_LOCK:
        ACTIVE_PROCESSES.discard(proc)


def handle_sigint(signum: int, frame: Any) -> None:
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()
    print("\nOperation cancelled by user.")
    with STATE_LOCK:
        for proc in list(ACTIVE_PROCESSES):
            with contextlib.suppress(OSError):
                proc.kill()
        for path in list(PART_PROGRESS):
            unlink_quiet(path)
    os._exit(130)


signal.signal(signal.SIGINT, handle_sigint)


@dataclass(frozen=True)
class PlexServer:
    name: str
    host: str
    port: int = 32400
    protocol: str = "http"
    token: str = ""
    user: str = ""

    @property
    def base_url(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}"


@dataclass(frozen=True)
class Track:
    rating_key: str
    title: str
    artist: str
    album: str
    album_artist: str
    parent_index: str
    index: str
    duration_ms: int
    media_url: str
    source_size: int
    playlist_id: str
    container: str = ""
    audio_codec: str = ""
    added_at: int = 0


@dataclass(frozen=True)
class DownloadJob:
    index: int
    total: int
    track: Track
    destination: Path


@dataclass(frozen=True)
class DownloadResult:
    job: DownloadJob
    success: bool
    skipped: bool
    bytes_written: int
    elapsed: float
    attempts: int
    error: str = ""


@dataclass(frozen=True)
class ExportOptions:
    """Fully resolved settings shared by every stage of an export run."""

    output_root: Path
    directory_limit: int
    conversion_formats: list[str]
    output_format: str
    quality: str
    max_workers: int
    retries: int
    retry_delay: float
    reserve_setting: Any
    reserve_bytes: int

    @classmethod
    def from_config(cls, config: dict) -> ExportOptions:
        output_root = Path(config["output"].get("directory", DOWNLOAD_DIR)).resolve()
        reserve_setting = config["output"].get("reserve", "5%")
        output_format, quality = conversion_spec(config)
        _, _, reserve_bytes = get_disk_stats(output_root, reserve_setting)
        return cls(
            output_root=output_root,
            directory_limit=safe_int(config["output"].get("directory_limit"), 255),
            conversion_formats=config["audio"].get("conversion_formats", ["aac:vbr"]),
            output_format=output_format,
            quality=quality,
            max_workers=conversion_threads(config),
            retries=safe_int(config["download"].get("retries"), 3),
            retry_delay=safe_float(config["download"].get("retry_delay"), 2.0),
            reserve_setting=reserve_setting,
            reserve_bytes=reserve_bytes,
        )


class AdaptiveConcurrency:
    """Halve the in-flight job count after failures, restoring it after a run of successes."""

    def __init__(self, maximum: int) -> None:
        self.stages = self._build_stages(max(ADAPTIVE_MIN_WORKERS, int(maximum)))
        self.stage = 0
        self.consecutive_successes = 0

    @staticmethod
    def _build_stages(maximum: int) -> list[int]:
        stages: list[int] = []
        value = maximum
        while value > 1:
            stages.append(value)
            value = max(1, value // 2)
        if not stages or stages[-1] != 1:
            stages.append(1)
        return stages

    @property
    def workers(self) -> int:
        return self.stages[self.stage]

    def failure(self) -> bool:
        self.consecutive_successes = 0
        if self.stage >= len(self.stages) - 1:
            return False
        self.stage += 1
        return True

    def success(self) -> bool:
        self.consecutive_successes += 1
        if self.consecutive_successes < ADAPTIVE_SUCCESS_THRESHOLD or self.stage <= 0:
            return False
        self.consecutive_successes = 0
        self.stage -= 1
        return True


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def unlink_quiet(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def parse_reserve(reserve_setting: Any, total_bytes: int) -> int:
    """Parses a reserve setting (e.g., '5%', '128M', '2G', '500K') into raw bytes."""
    if isinstance(reserve_setting, (int, float)):
        return int(reserve_setting)

    match = RESERVE_PATTERN.match(str(reserve_setting or "5%").strip())
    if match is None:
        return safe_int(reserve_setting)

    number, unit = safe_float(match.group(1)), (match.group(2) or "").upper()
    if unit == "%":
        return int(total_bytes * number / 100.0)
    return int(number * SIZE_UNITS.get(unit[:1], 1))


def resolve_existing_path(path: Path) -> Path:
    """Traverse up to the nearest existing parent directory on the filesystem."""
    target = path.resolve()
    while not target.exists() and target.parent != target:
        target = target.parent
    return target


def free_space(path: Path) -> int:
    target = resolve_existing_path(path)
    try:
        return shutil.disk_usage(target).free
    except OSError:
        return 0


def get_disk_stats(path: Path, reserve_setting: Any = "5%") -> tuple[int, int, int]:
    """Retrieve free space, total space, and calculated reserve bytes for an existing mount path."""
    target = resolve_existing_path(path)
    try:
        usage = shutil.disk_usage(target)
        reserve_bytes = parse_reserve(reserve_setting, usage.free)
        return usage.free, usage.total, reserve_bytes
    except OSError:
        return free_space(target), 0, 0


@functools.lru_cache(maxsize=4096)
def char_width(char: str) -> int:
    """Display columns for one character; variation selectors and ZWJ occupy none."""
    code = ord(char)
    if (0xFE00 <= code <= 0xFE0F) or (0xE0100 <= code <= 0xE01EF) or code == 0x200D:
        return 0
    return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1


def display_width(text: str) -> int:
    """Calculate terminal display column width using standard East Asian Width rules."""
    text = str(text)
    return len(text) if text.isascii() else sum(char_width(c) for c in text)


def truncate_to_width(text: str, max_width: int) -> str:
    """Truncate text strictly by display column width to prevent auto-wrapping."""
    text = str(text)
    if max_width <= 0:
        return ""
    if text.isascii():
        return text[:max_width]

    used = 0
    for position, char in enumerate(text):
        used += char_width(char)
        if used > max_width:
            return text[:position]
    return text


def pad_right(text: str, total_width: int) -> str:
    text = str(text)
    return text + " " * max(0, total_width - display_width(text))


_TERM_WIDTH_CACHE: tuple[float, int] = (0.0, 80)


def terminal_width(ttl: float = 0.5) -> int:
    """Terminal width behind a short TTL cache; the underlying query is too costly per frame."""
    global _TERM_WIDTH_CACHE
    now = time.monotonic()
    stamp, width = _TERM_WIDTH_CACHE
    if now - stamp >= ttl:
        width = shutil.get_terminal_size((80, 24)).columns
        _TERM_WIDTH_CACHE = (now, width)
    return width


def term_print(text: str = "", *, end: str = "\n", flush: bool = True) -> None:
    """Print line with absolute line clearing and padding to eliminate ghosting artifacts."""
    padded = pad_right(str(text), terminal_width() - 1)
    print(f"\r\033[K{padded}{end}", end="", flush=flush)


def human_size(value: int | float) -> str:
    if value < 1024:
        return f"{value:.1f} B"
    size = float(value)
    for unit in ("KB", "MB", "GB", "TB", "PB"):
        size /= 1024
        if size < 1024:
            return f"{size:.1f} {unit}"
    return f"{size:.1f} EB"


def human_rate(value: float) -> str:
    return f"{human_size(value)}/s"


def human_duration(milliseconds: Any) -> str:
    seconds = safe_int(milliseconds) // 1000
    if seconds <= 0:
        return "unknown"

    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"


@functools.lru_cache(maxsize=8192)
def sanitize_filename(value: str, fallback: str = "Unknown", max_bytes: int = 255) -> str:
    value = unicodedata.normalize("NFC", str(value or "")).replace("\x00", "")
    value = ILLEGAL_CHARS_PATTERN.sub("_", value)
    value = WHITESPACE_PATTERN.sub(" ", value).strip().rstrip(". ") or fallback

    if value.upper() in RESERVED_NAMES:
        value = f"_{value}"

    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value

    suffix = f"…{stable_hash(value, 6)}"
    budget = max_bytes - len(suffix.encode("utf-8"))
    if budget <= 0:
        return suffix.encode("utf-8")[:max_bytes].decode("utf-8", "ignore")
    return encoded[:budget].decode("utf-8", "ignore") + suffix


def stable_hash(value: str, length: int = 8) -> str:
    return hashlib.sha1(value.encode("utf-8"), usedforsecurity=False).hexdigest()[:length]


def canonical_format(value: str) -> str:
    """Collapse a container, codec, or extension spelling onto one canonical format name."""
    value = str(value or "").casefold().lstrip(".")
    return FORMAT_ALIASES.get(value, value)


def format_extension(fmt: str) -> str:
    canonical = canonical_format(fmt)
    return FORMAT_EXTENSIONS.get(canonical, f".{canonical}")


def track_filename(track: Track, number: int, fmt: str = "mp3") -> str:
    artist = sanitize_filename(track.artist, "Unknown Artist")
    album = sanitize_filename(track.album, "Unknown Album")
    title = sanitize_filename(track.title, "Unknown Track")

    num = safe_int(track.index) or number
    ext = format_extension(fmt)
    prefix_part = f"{num:02d} - {artist} - {album} - "
    available = 255 - len((prefix_part + ext).encode("utf-8"))

    if available <= 0:
        album = sanitize_filename(album, "Unknown Album", max_bytes=80)
        prefix_part = f"{num:02d} - {artist} - {album} - "
        available = 255 - len((prefix_part + ext).encode("utf-8"))

    title = sanitize_filename(title, "Unknown Track", max_bytes=max(1, available))
    return f"{prefix_part}{title}{ext}"


def deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        result[key] = deep_merge(result[key], value) if isinstance(result.get(key), dict) and isinstance(value, dict) else value
    return result


def save_config(path: Path, config: dict) -> None:
    temporary = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        config = deep_merge({}, DEFAULT_CONFIG)
        save_config(CONFIG_PATH, config)
        print(f"  Created {CONFIG_PATH}")
        return config

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {CONFIG_PATH}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise RuntimeError(f"Invalid configuration: {CONFIG_PATH}")

    return deep_merge(DEFAULT_CONFIG, loaded)


def http_get(url: str, token: str = "", timeout: int = 30) -> bytes:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/xml,application/json,text/plain,*/*"}
    if token:
        headers["X-Plex-Token"] = token
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def plex_xml(server: PlexServer, path: str, params: dict | None = None, timeout: int = 30) -> ET.Element:
    query = urllib.parse.urlencode(params or {})
    url = f"{server.base_url}{path}?{query}" if query else f"{server.base_url}{path}"
    
    try:
        return ET.fromstring(http_get(url, token=server.token, timeout=timeout))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Plex returned HTTP {exc.code} for {path}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Unable to reach Plex: {exc.reason}") from exc
    except ET.ParseError as exc:
        raise RuntimeError(f"Plex returned invalid XML for {path}") from exc


def test_server(server: PlexServer, timeout: int) -> ET.Element | None:
    try:
        return plex_xml(server, "/identity", timeout=timeout)
    except Exception:
        return None


def get_users(server: PlexServer, timeout: int = 30) -> list[str]:
    users: list[str] = []
    queries = [("/accounts", "Account", ("name", "title", "username")), ("/users", "User", ("title", "name", "username"))]

    for path, tag, attrs in queries:
        with contextlib.suppress(Exception):
            root = plex_xml(server, path, timeout=timeout)
            for el in root.findall(tag):
                name = next((el.attrib[a] for a in attrs if el.attrib.get(a)), "")
                if name and name not in users:
                    users.append(name)

    return users


def prompt_user(server: PlexServer, config: dict, path: Path = CONFIG_PATH) -> str:
    plex = config.get("plex", {})
    timeout = safe_int(plex.get("timeout"), 30)
    configured_user = str(plex.get("user") or "").strip()

    if configured_user:
        print(f"  User:    {configured_user}")
        return configured_user

    print("  Discovering Plex users…")
    users = get_users(server, timeout=timeout)

    if not users:
        selected_user = input("  Plex user (optional, press Enter to skip): ").strip()
    elif len(users) == 1:
        selected_user = users[0]
        print(f"  User:    {selected_user}")
    else:
        print("\n  Plex users:")
        for idx, u in enumerate(users, 1):
            print(f"    {idx}. {u}")
        while True:
            ans = input("\n  Select Plex user [1]: ").strip() or "1"
            if ans.isdigit() and 1 <= int(ans) <= len(users):
                selected_user = users[int(ans) - 1]
                break
            print("  Invalid selection.")
        print(f"  User:    {selected_user}")

    if selected_user:
        if "plex" not in config:
            config["plex"] = {}
        config["plex"]["user"] = selected_user
        save_config(path, config)
    return selected_user


def discover_gdm_servers(timeout: float = 3.0) -> list[PlexServer]:
    discovered: dict[tuple[str, int], PlexServer] = {}
    message = (
        b"M-SEARCH * HTTP/1.0\r\n"
        b"HOST: 239.0.0.250:32414\r\n"
        b'MAN: "ssdp:discover"\r\n'
        b"ST: plex/media-server\r\n\r\n"
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(0.5)

    try:
        sock.sendto(message, ("239.0.0.250", 32414))
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            try:
                data, address = sock.recvfrom(8192)
            except (socket.timeout, OSError):
                continue

            lines = data.decode("utf-8", errors="replace").splitlines()
            headers = {
                line.split(":", 1)[0].strip().lower(): line.split(":", 1)[1].strip()
                for line in lines[1:]
                if ":" in line
            }

            if "plex/media-server" not in headers.get("content-type", "").lower():
                continue

            host = headers.get("host", "").strip() or address[0]
            if host.endswith(".plex.direct"):
                host = address[0]

            port = safe_int(headers.get("port"), 32400)
            name = headers.get("name", host)
            discovered[(host, port)] = PlexServer(name=name, host=host, port=port)
    finally:
        sock.close()

    return list(discovered.values())


def prompt_server(config: dict, path: Path = CONFIG_PATH) -> PlexServer:
    plex = config["plex"]
    timeout = safe_int(plex.get("timeout"), 30)
    configured_host = str(plex.get("host") or "").strip()
    configured_port = safe_int(plex.get("port"), 32400)
    configured_token = str(plex.get("token") or "").strip()

    if configured_host:
        configured = PlexServer(name=configured_host, host=configured_host, port=configured_port, token=configured_token)
        if (identity := test_server(configured, timeout)) is not None:
            name = identity.attrib.get("friendlyName", configured_host)
            print(f"  Server:  {name}\n  Address: {configured.base_url}\n  Access:   OK")
            return PlexServer(name=name, host=configured_host, port=configured_port, token=configured_token)
        print(f"  Configured server unavailable: {configured.base_url}")

    print("  Discovering local Plex servers…")
    servers = discover_gdm_servers()

    if not servers:
        print("\n  No local Plex server discovered. Enter address manually or set plex.host in settings.json.\n")
        address = input("  Plex server address: ").strip()
        if not address:
            raise RuntimeError("No Plex server address supplied.")

        parsed = urllib.parse.urlparse(address)
        if parsed.scheme:
            protocol, host, port = parsed.scheme, parsed.hostname or address, parsed.port or configured_port
        else:
            protocol = "http"
            parts = address.rsplit(":", 1)
            host, port = (parts[0], int(parts[1])) if len(parts) == 2 and parts[1].isdigit() else (address, configured_port)
        servers = [PlexServer(name=host, host=host, port=port, protocol=protocol)]

    server = servers[0] if len(servers) == 1 else _select_server_interactive(servers)
    print(f"  Found: {server.name} ({server.base_url})")

    token = configured_token
    if test_server(server, timeout) is not None:
        print("  Local access: OK")
    else:
        print("\n  Local access requires authentication.")
        token = configured_token or input("  Plex token: ").strip()
        if not token:
            raise RuntimeError("A Plex token is required for this server.")

        authenticated = PlexServer(name=server.name, host=server.host, port=server.port, protocol=server.protocol, token=token)
        identity = test_server(authenticated, timeout)
        if identity is None:
            raise RuntimeError("Unable to connect to Plex with the supplied token.")

        print("  Authentication: OK")
        server = PlexServer(name=identity.attrib.get("friendlyName", server.name), host=server.host, port=server.port, protocol=server.protocol, token=token)

    config["plex"]["host"] = server.host
    config["plex"]["port"] = server.port
    config["plex"]["token"] = token
    save_config(path, config)
    return server


def _select_server_interactive(servers: list[PlexServer]) -> PlexServer:
    print("\n  Plex servers:")
    for idx, cand in enumerate(servers, 1):
        print(f"    {idx}. {cand.name} ({cand.base_url})")
    while True:
        ans = input("\n  Select Plex server [1]: ").strip() or "1"
        if ans.isdigit() and 1 <= int(ans) <= len(servers):
            return servers[int(ans) - 1]
        print("  Invalid selection.")


def select_music_library(server: PlexServer, timeout: int) -> tuple[str, str]:
    root = plex_xml(server, "/library/sections", timeout=timeout)
    libraries = [(d.attrib.get("key", ""), d.attrib.get("title", "Music")) for d in root.findall("Directory") if d.attrib.get("type") == "artist" and d.attrib.get("key")]

    if not libraries:
        raise RuntimeError("No Plex music library was found.")
    if len(libraries) == 1:
        return libraries[0]

    print("\nMusic libraries:\n")
    for idx, (_, title) in enumerate(libraries, 1):
        print(f"  {idx:>2}. {title}")

    while True:
        ans = input("\nSelect music library [1]: ").strip() or "1"
        if ans.isdigit() and 1 <= int(ans) <= len(libraries):
            return libraries[int(ans) - 1]
        print("Invalid selection.")


def get_playlists(server: PlexServer, timeout: int) -> list[tuple[str, str, int, str]]:
    root = plex_xml(server, "/playlists", params={"playlistType": "audio"}, timeout=timeout)
    return [
        (p.attrib["ratingKey"], p.attrib.get("title", "Unnamed"), safe_int(p.attrib.get("leafCount")), human_duration(p.attrib.get("duration")))
        for p in root.findall("Playlist") if "ratingKey" in p.attrib
    ]


def choose_playlists(playlists: list[tuple[str, str, int, str]], output_root: Path) -> list[tuple[str, str]]:
    if not playlists:
        raise RuntimeError("No Plex music playlists were found.")

    defaults = {p.name.casefold() for p in output_root.iterdir() if p.is_dir()} if output_root.is_dir() else set()
    print("\nMusic playlists:\n")

    for idx, (_, title, count, duration) in enumerate(playlists, 1):
        marker = "*" if sanitize_filename(title, "Music").casefold() in defaults else " "
        print(f"  {marker} {idx:>2}. {title} ({count:,} tracks, {duration})")

    print("     X. Random Fill Mode\n     A. All\n     Multiple selections: 1,3,5,X")

    default_indices = [idx for idx, (_, title, _, _) in enumerate(playlists, 1) if sanitize_filename(title, "Music").casefold() in defaults]
    default_random = "random" in defaults

    prompt_default = ",".join(map(str, default_indices))
    if default_random:
        prompt_default += ("," if prompt_default else "") + "X"
    prompt = f"\nSelect downloads [{prompt_default}]: " if prompt_default else "\nSelect downloads: "

    while True:
        ans = input(prompt).strip()
        if not ans and prompt_default:
            ans = prompt_default

        if ans.lower() == "a":
            return [(rk, title) for rk, title, _, _ in playlists]

        try:
            indices, random_selected = [], False
            for part in ans.split(","):
                part = part.strip()
                if not part:
                    continue
                if part.lower() == "x":
                    random_selected = True
                elif 1 <= int(part) <= len(playlists):
                    idx = int(part)
                    if idx not in indices:
                        indices.append(idx)
                else:
                    raise ValueError

            selected = [(playlists[i - 1][0], playlists[i - 1][1]) for i in indices]
            if random_selected:
                selected.append(("Random", "Random"))
            if selected:
                return selected
        except ValueError:
            pass
        print("Invalid selection.")


def safe_media_url(base_url: str, key: str) -> str:
    joined = urllib.parse.urljoin(base_url, key)
    parsed = urllib.parse.urlparse(joined)
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, urllib.parse.quote(parsed.path, safe="/%"), parsed.params, urllib.parse.quote(parsed.query, safe="=&?%"), parsed.fragment))


def track_from_xml(item: ET.Element, server: PlexServer, playlist_id: str) -> Track | None:
    media = item.find("Media")
    if media is None or (part := media.find("Part")) is None or not (key := part.attrib.get("key", "")):
        return None

    return Track(
        rating_key=item.attrib.get("ratingKey", ""),
        title=item.attrib.get("title", "Unknown Track"),
        artist=item.attrib.get("grandparentTitle", item.attrib.get("originalTitle", "Unknown Artist")),
        album=item.attrib.get("parentTitle", "Unknown Album"),
        album_artist=item.attrib.get("parentTitle", ""),
        parent_index=item.attrib.get("parentIndex", "1"),
        index=item.attrib.get("index", "0"),
        duration_ms=safe_int(item.attrib.get("duration")),
        media_url=safe_media_url(server.base_url, key),
        source_size=safe_int(part.attrib.get("size")),
        playlist_id=playlist_id,
        container=part.attrib.get("container", media.attrib.get("container", "")),
        audio_codec=part.attrib.get("audioCodec", media.attrib.get("audioCodec", "")),
        added_at=safe_int(item.attrib.get("addedAt")),
    )


def fetch_playlist_tracks(server: PlexServer, playlist_id: str, timeout: int) -> list[Track]:
    root = plex_xml(server, f"/playlists/{playlist_id}/items", timeout=timeout)
    return [t for item in root.findall("Track") if (t := track_from_xml(item, server, playlist_id)) is not None]


def fetch_library_tracks(server: PlexServer, library_key: str, timeout: int) -> list[Track]:
    root = plex_xml(server, f"/library/sections/{library_key}/all", params={"type": 10, "sort": "addedAt:desc"}, timeout=timeout)
    return [t for item in root.findall("Track") if (t := track_from_xml(item, server, "Random")) is not None]


def choose_directory_limit() -> int:
    print("\nDirectory limit\n  - Maximum 255 audio files per directory.\n  - Enter -1 for unlimited.\n")
    while True:
        ans = input("  Maximum files per directory [255]: ").strip() or "255"
        try:
            val = int(ans)
            if val == -1 or val > 0:
                return val
        except ValueError:
            pass
        print("  Enter a positive number or -1.")


def get_format_from_spec(spec: str) -> str:
    return spec.split(":", 1)[0].strip().lower()


def get_track_format(track: Track) -> str:
    """Canonical source format, trusting the container first, then the codec, then the URL."""
    url_extension = track.media_url.casefold().split("?", 1)[0].rpartition(".")[2]
    for candidate in (track.container, track.audio_codec, url_extension):
        canonical = canonical_format(candidate)
        if canonical in KNOWN_FORMATS:
            return canonical
    return canonical_format(track.container) or canonical_format(track.audio_codec) or "mp3"


def source_matches_output(track: Track, output_format: str) -> bool:
    return get_track_format(track) == canonical_format(output_format)


def is_format_supported(track: Track, conversion_formats: list[str]) -> bool:
    track_format = get_track_format(track)
    return any(track_format == canonical_format(get_format_from_spec(spec)) for spec in conversion_formats)


def build_output_path(options: ExportOptions, playlist_name: str, position: int, track: Track) -> Path:
    playlist_root = options.output_root / sanitize_filename(playlist_name, "Music")
    limit = options.directory_limit
    if limit == -1:
        directory, directory_position = playlist_root, position
    else:
        directory = playlist_root / f"{((position - 1) // limit) + 1:03d}"
        directory_position = ((position - 1) % limit) + 1

    fmt = (
        get_track_format(track)
        if is_format_supported(track, options.conversion_formats)
        else get_format_from_spec(options.conversion_formats[0])
    )
    return directory / track_filename(track, directory_position, fmt)


def conversion_spec(config: dict) -> tuple[str, str]:
    formats = config["audio"].get("conversion_formats", [])
    if not isinstance(formats, list) or not formats:
        raise RuntimeError("audio.conversion_formats must contain at least one format.")
    spec = str(formats[0]).strip()
    parts = spec.split(":", 1)
    output_format, quality = parts[0].strip().lower(), parts[1].strip() if len(parts) == 2 else ""
    if not FORMAT_NAME_PATTERN.fullmatch(output_format):
        raise RuntimeError(f"Invalid conversion format: {spec}")
    return output_format, quality


def conversion_threads(config: dict) -> int:
    val = config["audio"].get("conversion_threads", "auto")
    if isinstance(val, str) and val.strip().lower() == "auto":
        return os.cpu_count() or 1
    try:
        workers = int(val)
        if workers > 0:
            return workers
    except (TypeError, ValueError):
        pass
    raise RuntimeError('audio.conversion_threads must be "auto" or a positive integer.')


@functools.lru_cache(maxsize=None)
def check_program(program: str) -> bool:
    try:
        return subprocess.run([program, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0
    except OSError:
        return False


def ensure_ffmpeg() -> None:
    if not check_program("ffmpeg") or not check_program("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe are required for conversion but were not found in PATH.")


@functools.lru_cache(maxsize=1)
def has_libfdk_aac() -> bool:
    try:
        probe = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False)
    except OSError:
        return False
    return "libfdk_aac" in probe.stdout


def track_identity(track: Track) -> str:
    return "|".join((
        sanitize_filename(track.artist, "Unknown Artist").casefold(),
        sanitize_filename(track.album, "Unknown Album").casefold(),
        sanitize_filename(track.title, "Unknown Track").casefold(),
    ))


def parse_file_identity(path: Path) -> str | None:
    parts = [segment.strip() for segment in path.stem.split(" - ", 3)]
    if len(parts) == 4:
        parts = parts[1:]
    elif len(parts) != 3:
        return None
    return "|".join(segment.casefold() for segment in parts)


def verify_track_duration(temp_path: Path, track: Track) -> None:
    """Verify audio stream integrity and duration post-download or conversion using ffprobe."""
    if track.duration_ms <= 0:
        return

    expected_seconds = track.duration_ms / 1000.0
    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(temp_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        actual_seconds = float(probe.stdout.strip()) if probe.returncode == 0 else 0.0
    except (OSError, ValueError):
        return  # An unreadable probe is not evidence of truncation; accept the file.

    if actual_seconds and actual_seconds < (expected_seconds - DURATION_TOLERANCE_SECONDS):
        raise RuntimeError(f"File truncated: expected ~{expected_seconds:.1f}s, got {actual_seconds:.1f}s.")


def iter_audio_files(root: Path) -> Iterator[Path]:
    """Yield audio files under root, testing the cheap suffix before touching the filesystem."""
    for path in root.rglob("*.*"):
        if path.suffix.lower() in AUDIO_EXTENSIONS and path.is_file():
            yield path


def prune_empty_dirs(root: Path) -> None:
    for dirpath, _, _ in os.walk(root, topdown=False):
        directory = Path(dirpath)
        if directory != root and not any(directory.iterdir()):
            with contextlib.suppress(OSError):
                directory.rmdir()


def cleanup_playlist_leftovers(options: ExportOptions, playlist_name: str, tracks: list[Track]) -> None:
    playlist_root = options.output_root / sanitize_filename(playlist_name, "Music")
    if not playlist_root.exists():
        return

    expected_paths = {
        build_output_path(options, playlist_name, position, track)
        for position, track in enumerate(tracks, 1)
    }

    for path in iter_audio_files(playlist_root):
        if path not in expected_paths:
            unlink_quiet(path)

    prune_empty_dirs(playlist_root)


def cleanup_random_fill_leftovers(
    output_root: Path,
    all_library_tracks: list[Track],
    max_random_tracks: int = 1000,
    excluded_identities: set[str] | None = None,
) -> None:
    random_root = output_root / "Random"
    if not random_root.exists():
        return

    valid_keys = {track_identity(t) for t in all_library_tracks}
    if excluded_identities:
        valid_keys -= excluded_identities

    existing_files: list[tuple[Path, float]] = []
    for path in iter_audio_files(random_root):
        key = parse_file_identity(path)
        if not key or key not in valid_keys:
            unlink_quiet(path)
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        existing_files.append((path, mtime))

    if max_random_tracks > 0 and len(existing_files) > max_random_tracks:
        existing_files.sort(key=lambda entry: entry[1])
        for path, _ in existing_files[: len(existing_files) - max_random_tracks]:
            unlink_quiet(path)

    prune_empty_dirs(random_root)


def get_existing_random_identities_and_count(output_root: Path) -> tuple[set[str], int]:
    random_root = output_root / "Random"
    if not random_root.exists():
        return set(), 0

    existing_keys: set[str] = set()
    count = 0
    for path in iter_audio_files(random_root):
        count += 1
        if key := parse_file_identity(path):
            existing_keys.add(key)
    return existing_keys, count


def select_random_tracks(
    tracks: list[Track],
    output_root: Path,
    max_random_tracks: int = 1000,
    strategy: str = "freshness",
    excluded_identities: set[str] | None = None,
) -> tuple[list[Track], int]:
    cleanup_random_fill_leftovers(output_root, tracks, max_random_tracks=max_random_tracks, excluded_identities=excluded_identities)

    existing_keys, existing_count = get_existing_random_identities_and_count(output_root)
    if excluded_identities:
        existing_keys.update(excluded_identities)

    candidates = [t for t in tracks if track_identity(t) not in existing_keys]

    if strategy == "freshness":
        candidates.sort(key=lambda t: t.added_at, reverse=True)
    else:
        random.SystemRandom().shuffle(candidates)

    if max_random_tracks > 0:
        current_total_random = existing_count - 1 + len(candidates)
        if current_total_random > max_random_tracks:
            allowed_new = max(0, max_random_tracks - (existing_count - 1))
            candidates = candidates[:allowed_new]

    return candidates, existing_count + 1


def ffmpeg_command(track: Track, output: Path, token: str, output_format: str, quality: str) -> list[str]:
    output_format = output_format.lower()
    norm_q = quality.upper() if quality else "VBR"

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-threads", "1",
        "-reconnect", "1", "-reconnect_streamed", "1",
        "-reconnect_on_network_error", "1",
        "-reconnect_on_http_error", "429,500,502,503,504",
        "-reconnect_delay_max", "5"
    ]
    if token:
        cmd.extend(["-headers", f"X-Plex-Token: {token}\r\n"])
    cmd.extend(["-i", track.media_url, "-vn", "-map", "0:a:0", "-map_metadata", "0"])

    if output_format in ("aac", "m4a"):
        if has_libfdk_aac():
            cmd.extend(["-c:a", "libfdk_aac"])
            cmd.extend(["-b:a", norm_q.lower()] if BITRATE_PATTERN.fullmatch(norm_q) else ["-vbr", "5"])
        else:
            bitrate = norm_q.lower() if BITRATE_PATTERN.fullmatch(norm_q) else "320k"
            cmd.extend(["-c:a", "aac", "-b:a", bitrate])
        cmd.extend(["-f", "mp4"])
    elif output_format == "mp3":
        cmd.extend(["-map", "0:v?", "-c:v", "copy", "-id3v2_version", "3", "-c:a", "libmp3lame"])
        cmd.extend(["-b:a", norm_q.lower()] if BITRATE_PATTERN.fullmatch(norm_q) else ["-q:a", "0"])
        cmd.extend(["-f", "mp3"])
    elif output_format == "flac":
        cmd.extend(["-c:a", "flac", "-f", "flac"])
    elif output_format in ("ogg", "opus", "vorbis"):
        codec = "libopus" if output_format == "opus" else "libvorbis"
        cmd.extend(["-c:a", codec, "-f", "ogg"])
    else:
        # The .part suffix hides the real extension, so the muxer must be named explicitly.
        cmd.extend(["-c:a", "copy", "-f", output_format])

    cmd.append(str(output))
    return cmd


def download_url(media_url: str) -> str:
    """Plex only serves the untouched original when asked to download; otherwise it may remux."""
    return f"{media_url}{'&' if '?' in media_url else '?'}download=1"


def expected_response_bytes(response: Any, resuming: bool, existing_size: int) -> int:
    """Total file size the server committed to, which outranks Plex's cached metadata size."""
    content_range = response.headers.get("Content-Range", "")
    if "/" in content_range:
        if (total := safe_int(content_range.rsplit("/", 1)[1])) > 0:
            return total

    length = safe_int(response.headers.get("Content-Length"))
    if length <= 0:
        return 0
    return length + existing_size if resuming else length


def _finalize_part(job: DownloadJob, temp_path: Path, written: int, expected: int = 0) -> int:
    """Validate a finished .part file and atomically promote it to its destination."""
    if written == 0:
        unlink_quiet(temp_path)
        raise RuntimeError("Downloaded file is empty.")

    if expected > 0 and written != expected:
        unlink_quiet(temp_path)
        raise RuntimeError(f"Download truncated: expected {expected:,} bytes, got {written:,} bytes.")

    try:
        verify_track_duration(temp_path, job.track)
    except RuntimeError:
        unlink_quiet(temp_path)  # Discard so a retry refetches instead of resuming bad bytes.
        raise

    os.replace(temp_path, job.destination)
    return written


def download_direct(job: DownloadJob, token: str, timeout: int = 30) -> int:
    temp_path = part_path(job.destination)
    job.destination.parent.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["X-Plex-Token"] = token

    source_size = job.track.source_size
    existing_size = temp_path.stat().st_size if temp_path.exists() else 0
    if source_size > 0 and existing_size > source_size:
        unlink_quiet(temp_path)  # A part longer than the source can only be corrupt.
        existing_size = 0
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"

    _register_part_file(temp_path, existing_size)
    try:
        request = urllib.request.Request(download_url(job.track.media_url), headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = getattr(response, "status", None) or getattr(response, "code", 200)
            # Anything other than a 206 means the server restarted the stream from zero.
            resuming = existing_size > 0 and status_code == 206
            bytes_written = existing_size if resuming else 0
            expected_total = expected_response_bytes(response, resuming, existing_size)

            with temp_path.open("ab" if resuming else "wb") as handle:
                while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                    handle.write(chunk)
                    bytes_written += len(chunk)
                    _update_part_progress(temp_path, bytes_written)

        return _finalize_part(job, temp_path, bytes_written, expected=expected_total)
    except urllib.error.HTTPError as exc:
        # 416 means the range started at or past EOF: the part already holds the whole file.
        if exc.code == 416 and source_size > 0 and temp_path.exists() and temp_path.stat().st_size == source_size:
            return _finalize_part(job, temp_path, source_size, expected=source_size)
        raise RuntimeError(f"HTTP {exc.code} {exc.reason} for {job.destination.name}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error: {exc.reason}") from exc
    finally:
        _unregister_part_file(temp_path)


def convert_track(job: DownloadJob, token: str, output_format: str, quality: str) -> int:
    temp_path = part_path(job.destination)
    job.destination.parent.mkdir(parents=True, exist_ok=True)
    unlink_quiet(temp_path)

    _register_part_file(temp_path)  # Size is unknown up front, so it gets polled instead.
    command = ffmpeg_command(job.track, temp_path, token, output_format, quality)
    try:
        proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        _register_process(proc)
        try:
            _, stderr_data = proc.communicate()
        finally:
            _unregister_process(proc)

        if proc.returncode != 0 or not temp_path.exists() or temp_path.stat().st_size == 0:
            unlink_quiet(temp_path)
            raise RuntimeError((stderr_data or "").strip() or "FFmpeg conversion failed")

        return _finalize_part(job, temp_path, temp_path.stat().st_size)
    finally:
        _unregister_part_file(temp_path)


def _reuse_existing(job: DownloadJob, is_direct: bool) -> bool:
    """Report whether the destination already holds a usable file, deleting it if not."""
    if not job.destination.exists():
        return False

    # Only a short file proves an interrupted run; Plex's recorded size can lag the real one.
    if is_direct and job.track.source_size > 0:
        if job.destination.stat().st_size >= job.track.source_size:
            return True
        unlink_quiet(job.destination)
        return False

    return job.destination.stat().st_size > 0


def download_track(job: DownloadJob, token: str, options: ExportOptions) -> DownloadResult:
    # Direct-copy when the source is already in a configured format; this must
    # mirror build_output_path's format choice so the .part/resume path is used
    # for every supported format, not only the top-priority one.
    is_direct = is_format_supported(job.track, options.conversion_formats)

    if _reuse_existing(job, is_direct):
        return DownloadResult(job=job, success=True, skipped=True, bytes_written=0, elapsed=0.0, attempts=0)

    start_time = time.monotonic()
    last_error = ""

    for attempt in range(1, options.retries + 1):
        try:
            bytes_written = (
                download_direct(job, token)
                if is_direct
                else convert_track(job, token, options.output_format, options.quality)
            )
            return DownloadResult(
                job=job,
                success=True,
                skipped=False,
                bytes_written=bytes_written,
                elapsed=max(0.001, time.monotonic() - start_time),
                attempts=attempt,
            )
        except Exception as exc:
            last_error = str(exc)
            if attempt < options.retries:
                time.sleep(options.retry_delay * attempt)

    return DownloadResult(
        job=job,
        success=False,
        skipped=False,
        bytes_written=0,
        elapsed=max(0.001, time.monotonic() - start_time),
        attempts=options.retries,
        error=last_error,
    )


def render_progress_bar(completed: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return " " * width
    filled = max(0, min(width, int(round(width * completed / total))))
    return "█" * filled + "░" * (width - filled)


def process_download_queue(
    jobs: list[DownloadJob],
    token: str,
    options: ExportOptions,
    *,
    is_random_fill: bool = False,
) -> tuple[list[DownloadResult], bool]:
    if not jobs:
        return [], False

    max_workers = options.max_workers
    reserve_bytes = options.reserve_bytes
    output_root = options.output_root

    adaptive = AdaptiveConcurrency(max_workers)
    results: list[DownloadResult] = []
    completed = 0
    skipped_count = 0
    total = len(jobs)
    start_time = time.monotonic()
    total_bytes = 0
    stopped_on_reserve = False

    active_jobs_map: dict[int, str] = {}
    active_lock = threading.Lock()

    term_print(f"\nProcessing {total:,} track(s) using up to {max_workers} worker(s)…" if not is_random_fill else f"\nProcessing Random Fill tracks (target reserve: {human_size(reserve_bytes)}) using up to {max_workers} worker(s)…")

    def _execute_job(j: DownloadJob) -> DownloadResult:
        thread_id = threading.get_ident()
        track_desc = truncate_to_width(f"{j.track.artist} - {j.track.title}", 25)
        with active_lock:
            active_jobs_map[thread_id] = track_desc
        try:
            return download_track(j, token, options)
        finally:
            with active_lock:
                active_jobs_map.pop(thread_id, None)

    print_lock = threading.Lock()

    last_render_time = time.monotonic()
    last_scan_time = 0.0
    last_total_bytes = 0
    scanned_active_bytes = 0
    scanned_free_space = 0
    ewma_rate = 0.0
    min_render_interval = 0.04
    scan_interval = 0.1
    projecting = is_random_fill and reserve_bytes > 0

    with print_lock:
        sys.stdout.write("\n\n")
        sys.stdout.write("\033[2A")
        sys.stdout.write("\033[s")
        sys.stdout.flush()

    def _average_track_bytes() -> float:
        downloaded_count = completed - skipped_count
        if downloaded_count > 0 and total_bytes > 0:
            return total_bytes / downloaded_count
        sample = jobs[:10]
        estimate = sum(j.track.source_size for j in sample) / len(sample) if sample else 0
        return estimate or 8 * 1024 * 1024

    def _format_eta(seconds: float) -> str:
        return human_duration(int(seconds * 1000))

    def render_progress(force: bool = False) -> None:
        nonlocal last_render_time, last_scan_time, last_total_bytes
        nonlocal scanned_active_bytes, scanned_free_space, ewma_rate

        now = time.monotonic()
        dt = now - last_render_time
        if not force and dt < min_render_interval:
            return

        # Filesystem probes are far costlier than a repaint, so they run on a slower cadence.
        if force or now - last_scan_time >= scan_interval:
            scanned_active_bytes = active_part_bytes()
            scanned_free_space = free_space(output_root) if projecting else 0
            last_scan_time = now

        current_total = total_bytes + scanned_active_bytes

        if dt >= scan_interval:
            delta = current_total - last_total_bytes
            if delta >= 0:
                ewma_rate = 0.2 * (delta / dt) + 0.8 * ewma_rate
                last_total_bytes = current_total
            last_render_time = now

        elapsed = max(0.001, now - start_time)
        free_above_reserve = max(0, scanned_free_space - reserve_bytes) if projecting else 0

        # Random fill runs until the disk reserve is hit, so its total is a moving projection.
        display_total = total
        if projecting:
            estimated_remaining = int(free_above_reserve / _average_track_bytes())
            display_total = max(completed, min(len(jobs), completed + estimated_remaining))

        pct = (completed / display_total) * 100 if display_total > 0 else 0

        eta_str = "unknown"
        if projecting and ewma_rate > 0 and free_above_reserve > 0:
            eta_str = _format_eta(free_above_reserve / ewma_rate)
        elif ewma_rate > 0 and completed > 0:
            eta_str = _format_eta((display_total - completed) * (elapsed / completed))

        safe_width = max(10, terminal_width() - 2)
        bar_str = render_progress_bar(completed, display_total, width=12 if safe_width < 60 else 20)
        skip_str = f" ({skipped_count:,} skip)" if skipped_count > 0 else ""

        with active_lock:
            current_active = list(active_jobs_map.values())

        line1_raw = f" {bar_str} {pct:5.1f}% {completed}/{display_total}{skip_str} | {human_size(current_total)} | {human_rate(ewma_rate)} | ETA: {eta_str}"
        line1 = pad_right(truncate_to_width(line1_raw, safe_width), safe_width)

        line2 = ""
        if current_active:
            active_desc = "; ".join(current_active[:3]) + ("…" if len(current_active) > 3 else "")
            line2 = pad_right(truncate_to_width(f"Active: {active_desc}", safe_width), safe_width)

        with print_lock:
            sys.stdout.write("\033[u")
            sys.stdout.write(f"\033[K{line1}\n\033[K{line2}")
            sys.stdout.flush()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        job_iter = iter(jobs)
        futures: dict[concurrent.futures.Future, DownloadJob] = {}

        def _submit_next() -> bool:
            nonlocal stopped_on_reserve
            try:
                job = next(job_iter)
            except StopIteration:
                return False

            if reserve_bytes > 0:
                # In-flight parts have not landed on disk yet; hold back a slot for each.
                active_buffer = active_part_count() * (4 * 1024 * 1024)
                expected_size = job.track.source_size or (8 * 1024 * 1024)
                if (free_space(job.destination.parent) - active_buffer - expected_size) <= reserve_bytes:
                    stopped_on_reserve = True
                    return False

            futures[executor.submit(_execute_job, job)] = job
            return True

        def _fill_slots() -> None:
            while not stopped_on_reserve and len(futures) < adaptive.workers and _submit_next():
                pass

        _fill_slots()

        while futures:
            done, _ = concurrent.futures.wait(futures.keys(), timeout=0.05, return_when=concurrent.futures.FIRST_COMPLETED)
            if not done:
                render_progress()
                continue

            for fut in done:
                res = fut.result()
                del futures[fut]
                results.append(res)
                completed += 1

                if res.skipped:
                    skipped_count += 1
                elif res.success:
                    total_bytes += res.bytes_written
                    adaptive.success()
                else:
                    adaptive.failure()

            render_progress()
            _fill_slots()

    render_progress(force=True)

    if skipped_count == total:
        term_print(f"✓ All {total:,} tracks are already up to date.")
    else:
        term_print(f"✓ Processed {completed:,} tracks ({human_size(total_bytes)} downloaded, {skipped_count:,} skipped).")

    failures = [r for r in results if not r.success]
    if failures:
        term_print(f"✗ {len(failures):,} track(s) failed:")
        for res in failures[:10]:
            label = truncate_to_width(f"{res.job.track.artist} - {res.job.track.title}", 40)
            term_print(f"    {label}: {res.error or 'unknown error'}")
        if len(failures) > 10:
            term_print(f"    … and {len(failures) - 10:,} more.")

    return results, stopped_on_reserve


def connect_to_plex(config: dict) -> tuple[PlexServer, int]:
    print("\n--- Plex Connection ---")
    server = prompt_server(config, CONFIG_PATH)
    user = prompt_user(server, config, CONFIG_PATH)
    server = replace(server, user=user)
    return server, safe_int(config["plex"].get("timeout"), 30)


def build_playlist_jobs(
    server: PlexServer,
    selected_playlists: list[tuple[str, str]],
    options: ExportOptions,
    timeout: int,
) -> tuple[list[DownloadJob], bool]:
    jobs: list[DownloadJob] = []
    has_random = False

    for rating_key, title in selected_playlists:
        if rating_key == "Random":
            has_random = True
            continue

        term_print(f"Fetching tracks for playlist: {title}…")
        tracks = fetch_playlist_tracks(server, rating_key, timeout=timeout)
        cleanup_playlist_leftovers(options, title, tracks)

        jobs.extend(
            DownloadJob(
                index=position,
                total=len(tracks),
                track=track,
                destination=build_output_path(options, title, position, track),
            )
            for position, track in enumerate(tracks, 1)
        )

    return jobs, has_random


def build_random_jobs(
    server: PlexServer,
    library_key: str,
    options: ExportOptions,
    config: dict,
    timeout: int,
    excluded_identities: set[str],
) -> list[DownloadJob]:
    term_print("\nFetching tracks for library (Random Fill)…")
    library_tracks = fetch_library_tracks(server, library_key, timeout=timeout)

    random_config = config.get("random", {})
    candidates, start_position = select_random_tracks(
        library_tracks,
        options.output_root,
        max_random_tracks=safe_int(random_config.get("max_tracks"), 1000),
        strategy=str(random_config.get("strategy", "freshness")),
        excluded_identities=excluded_identities,
    )

    (options.output_root / "Random").mkdir(parents=True, exist_ok=True)

    total = len(candidates) + start_position - 1
    return [
        DownloadJob(
            index=position,
            total=total,
            track=track,
            destination=build_output_path(options, "Random", position, track),
        )
        for position, track in enumerate(candidates, start_position)
    ]


def print_summary(results: list[DownloadResult], options: ExportOptions, stopped_on_reserve: bool) -> None:
    successes = [r for r in results if r.success and not r.skipped]
    skipped = [r for r in results if r.skipped]
    failures = [r for r in results if not r.success]

    total_bytes = sum(r.bytes_written for r in successes)
    total_time = sum(r.elapsed for r in results)

    print("\n--- Summary ---")
    print(f"  Downloaded:         {len(successes):,} tracks ({human_size(total_bytes)})")
    print(f"  Skipped:            {len(skipped):,} tracks (already present)")
    print(f"  Failed:             {len(failures):,} tracks")
    if stopped_on_reserve:
        print(f"  Stopped on Reserve: Yes ({options.reserve_setting} / {human_size(options.reserve_bytes)} safety limit reached)")
    if total_time > 0:
        print(f"  Total time:         {human_duration(int(total_time * 1000))}")
    print("\nDone.")


def main() -> None:
    print(f"=== {APP_NAME} ===")
    config = load_config()

    ensure_ffmpeg()

    server, timeout = connect_to_plex(config)

    print("\n--- Music Library ---")
    library_key, library_title = select_music_library(server, timeout=timeout)
    print(f"  Selected: {library_title}\n")

    options = ExportOptions.from_config(config)
    playlists = get_playlists(server, timeout=timeout)
    selected_playlists = choose_playlists(playlists, options.output_root)

    if options.directory_limit not in (-1, 255):
        options = replace(options, directory_limit=choose_directory_limit())

    all_jobs, has_random = build_playlist_jobs(server, selected_playlists, options, timeout)
    results, stopped_on_reserve = process_download_queue(all_jobs, server.token, options)

    if has_random and not stopped_on_reserve:
        random_jobs = build_random_jobs(
            server,
            library_key,
            options,
            config,
            timeout,
            excluded_identities={track_identity(job.track) for job in all_jobs},
        )
        if random_jobs:
            random_results, random_stopped = process_download_queue(
                random_jobs, server.token, options, is_random_fill=True
            )
            results.extend(random_results)
            stopped_on_reserve = stopped_on_reserve or random_stopped

    print_summary(results, options, stopped_on_reserve)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
