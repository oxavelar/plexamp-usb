#!/usr/bin/env python3
"""Export Plex music to a car-friendly USB filesystem.

The program discovers a reachable Plex Media Server, authenticates only when
local unauthenticated access is unavailable, lets the user select Plex music
playlists, and exports tracks directly below a Downloads directory beside
this script.

Supported source formats are copied directly when they match any format in
the configured conversion/supported list. Other formats are converted using
FFmpeg exclusively to the top choice in order (defaulting to AAC-LC VBR
highest quality).

Downloads are deterministic and resumable. Existing complete files are kept;
incomplete, invalid, or mismatched files are replaced and retried.

Directories can be capped at a user-selected number of audio files. A value
of -1 disables the limit.

A playlist named "Random" is treated as a special fill mode. Tracks are
sampled from the complete Plex music library and downloaded until the
destination filesystem reaches its safety threshold (1% of total capacity).

Configuration is stored in settings.json beside this script.

Requirements:
    - Python 3.10+
    - ffmpeg and ffprobe when conversion is required
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import unicodedata


APP_NAME = "plexamp-usb"
CONFIG_JSON = "settings.json"
DOWNLOAD_DIR = "Downloads"

DURATION_TOLERANCE_SECONDS = 2.0
ADAPTIVE_SUCCESS_THRESHOLD = 8
ADAPTIVE_MIN_WORKERS = 1

DEFAULT_CONFIG = {
    "plex": {
        "host": "",
        "port": 32400,
        "token": "",
        "timeout": 30,
    },
    "audio": {
        "conversion_formats": ["aac:vbr", "mp3:vbr"],
        "conversion_threads": "auto",
    },
    "output": {
        "directory": DOWNLOAD_DIR,
        "directory_limit": 255,
    },
    "download": {
        "retries": 3,
        "retry_delay": 2.0,
    },
}

_RANDOM_INITIAL_FREE = 0
_RANDOM_TARGET_FILL = 0
_HAS_FDK_AAC: bool | None = None


@dataclass(frozen=True)
class PlexServer:
    name: str
    host: str
    port: int = 32400
    protocol: str = "http"
    token: str = ""

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


class AdaptiveConcurrency:
    """Dynamically adjust download concurrency."""

    def __init__(self, maximum: int) -> None:
        self.maximum = max(ADAPTIVE_MIN_WORKERS, int(maximum))
        self.stage = 0
        self.consecutive_successes = 0

    def _stages(self) -> list[int]:
        stages: list[int] = []
        value = self.maximum
        while value > 1:
            stages.append(value)
            value = max(1, value // 2)
        if not stages or stages[-1] != 1:
            stages.append(1)
        return stages

    @property
    def workers(self) -> int:
        stages = self._stages()
        self.stage = min(self.stage, len(stages) - 1)
        return stages[self.stage]

    def failure(self) -> bool:
        self.consecutive_successes = 0
        stages = self._stages()
        if self.stage >= len(stages) - 1:
            return False
        self.stage += 1
        return True

    def success(self) -> bool:
        self.consecutive_successes += 1
        if self.consecutive_successes < ADAPTIVE_SUCCESS_THRESHOLD:
            return False
        self.consecutive_successes = 0
        if self.stage <= 0:
            return False
        self.stage -= 1
        return True


def safe_int(value: str | int | None, default: int = 0) -> int:
    try:
        return int(float(value or default))
    except (TypeError, ValueError):
        return default


def unlink_quiet(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def display_width(text: str) -> int:
    """Calculate terminal display column width using standard East Asian Width rules."""
    width = 0
    for char in str(text):
        code = ord(char)
        if (0xFE00 <= code <= 0xFE0F) or (0xE0100 <= code <= 0xE01EF) or (code == 0x200D):
            continue
        ea = unicodedata.east_asian_width(char)
        if ea in ('W', 'F'):
            width += 2
        else:
            width += 1
    return width


def pad_right(text: str, total_width: int) -> str:
    text_str = str(text)
    w = display_width(text_str)
    if w >= total_width:
        return text_str
    return text_str + " " * (total_width - w)


def compact(text: str, width: int) -> str:
    text = str(text)
    if display_width(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    
    current_w = 0
    result = []
    for char in text:
        code = ord(char)
        char_w = 0 if ((0xFE00 <= code <= 0xFE0F) or (0xE0100 <= code <= 0xE01EF) or (code == 0x200D)) else (2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1)
        if current_w + char_w + 1 > width:
            break
        result.append(char)
        current_w += char_w
    return "".join(result) + "…"


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


def human_duration(milliseconds: int | str | None) -> str:
    seconds = safe_int(milliseconds) // 1000
    if seconds <= 0:
        return "unknown"

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def sanitize_filename(value: str, fallback: str = "Unknown", max_bytes: int = 255) -> str:
    value = unicodedata.normalize("NFC", str(value or "")).replace("\x00", "")
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]', "_", value)
    value = re.sub(r"\s+", " ", value).strip().rstrip(". ")

    if not value:
        value = fallback

    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
    if value.upper() in reserved:
        value = f"_{value}"

    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value

    suffix = f"…{stable_hash(value, 6)}"
    while value:
        candidate = value + suffix
        if len(candidate.encode("utf-8")) <= max_bytes:
            return candidate
        value = value[:-1]

    return suffix.encode("utf-8")[:max_bytes].decode("utf-8", "ignore")


def stable_hash(value: str, length: int = 8) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def format_extension(fmt: str) -> str:
    fmt = fmt.lower()
    if fmt in ("aac", "m4a", "mp4"):
        return ".m4a"
    if fmt == "mp3":
        return ".mp3"
    if fmt == "flac":
        return ".flac"
    if fmt in ("ogg", "opus", "vorbis"):
        return ".ogg"
    return f".{fmt}"


def track_filename(track: Track, number: int, fmt: str = "mp3") -> str:
    artist = sanitize_filename(track.artist, "Unknown Artist")
    album = sanitize_filename(track.album, "Unknown Album")
    title = sanitize_filename(track.title, "Unknown Track")

    track_number = safe_int(track.index)
    prefix = f"{track_number:02d}" if track_number else f"{number:02d}"

    prefix_part = f"{prefix} - {artist} - {album} - "
    ext = format_extension(fmt)
    available = 255 - len((prefix_part + ext).encode("utf-8"))

    if available <= 0:
        album = sanitize_filename(album, "Unknown Album", max_bytes=80)
        prefix_part = f"{prefix} - {artist} - {album} - "
        available = 255 - len((prefix_part + ext).encode("utf-8"))

    title = sanitize_filename(title, "Unknown Track", max_bytes=max(1, available))
    return f"{prefix_part}{title}{ext}"


def deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def save_config(path: Path, config: dict) -> None:
    temporary = path.with_name(path.name + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def load_config() -> dict:
    path = Path(__file__).resolve().parent / CONFIG_JSON
    if not path.exists():
        config = deep_merge({}, DEFAULT_CONFIG)
        save_config(path, config)
        print(f"  Created {path}")
        return config

    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise RuntimeError(f"Invalid configuration: {path}")

    return deep_merge(DEFAULT_CONFIG, loaded)


def http_get(url: str, token: str = "", timeout: int = 30) -> bytes:
    headers = {"User-Agent": f"{APP_NAME}/1.0", "Accept": "application/xml,application/json,text/plain,*/*"}
    if token:
        headers["X-Plex-Token"] = token
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def plex_xml(server: PlexServer, path: str, params: dict | None = None, timeout: int = 30) -> ET.Element:
    query = urllib.parse.urlencode(params or {})
    url = f"{server.base_url}{path}?{query}" if query else f"{server.base_url}{path}"
    
    try:
        payload = http_get(url, token=server.token, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Plex returned HTTP {exc.code} for {path}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Unable to reach Plex: {exc.reason}") from exc

    try:
        return ET.fromstring(payload)
    except ET.ParseError as exc:
        raise RuntimeError(f"Plex returned invalid XML for {path}") from exc


def test_server(server: PlexServer, timeout: int) -> ET.Element | None:
    try:
        return plex_xml(server, "/identity", timeout=timeout)
    except Exception:
        return None


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
            except socket.timeout:
                continue
            except OSError:
                break

            lines = data.decode("utf-8", errors="replace").splitlines()
            if not lines:
                continue

            headers = dict(
                (parts[0].strip().lower(), parts[1].strip())
                for line in lines[1:] if ":" in line and (parts := line.split(":", 1))
            )

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


def prompt_server(config: dict) -> PlexServer:
    plex = config["plex"]
    timeout = safe_int(plex.get("timeout"), 30)
    configured_host = str(plex.get("host") or "").strip()
    configured_port = safe_int(plex.get("port"), 32400)
    configured_token = str(plex.get("token") or "").strip()

    if configured_host:
        configured = PlexServer(name=configured_host, host=configured_host, port=configured_port, token=configured_token)
        if (identity := test_server(configured, timeout)) is not None:
            name = identity.attrib.get("friendlyName", configured_host)
            print(f"  Server: {name}\n  Address: {configured.base_url}\n  Access:  OK")
            return PlexServer(name=name, host=configured_host, port=configured_port, token=configured_token)
        print(f"  Configured server unavailable: {configured.base_url}")

    print("  Discovering local Plex servers...")
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

    if test_server(server, timeout) is not None:
        print("  Local access: OK")
        return server

    print("\n  Local access requires authentication.")
    token = configured_token or input("  Plex token: ").strip()
    if not token:
        raise RuntimeError("A Plex token is required for this server.")

    authenticated = PlexServer(name=server.name, host=server.host, port=server.port, protocol=server.protocol, token=token)
    identity = test_server(authenticated, timeout)
    if identity is None:
        raise RuntimeError("Unable to connect to Plex with the supplied token.")

    print("  Authentication: OK")
    return PlexServer(name=identity.attrib.get("friendlyName", server.name), host=server.host, port=server.port, protocol=server.protocol, token=token)


def _select_server_interactive(servers: list[PlexServer]) -> PlexServer:
    print("\n  Plex servers:")
    for idx, cand in enumerate(servers, 1):
        print(f"    {idx}. {cand.name} ({cand.base_url})")
    while True:
        ans = input("\n  Select Plex server [1]: ").strip()
        if not ans:
            return servers[0]
        try:
            idx = int(ans)
            if 1 <= idx <= len(servers):
                return servers[idx - 1]
        except ValueError:
            pass
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
        ans = input("\nSelect music library [1]: ").strip()
        if not ans:
            return libraries[0]
        try:
            idx = int(ans)
            if 1 <= idx <= len(libraries):
                return libraries[idx - 1]
        except ValueError:
            pass
        print("Invalid selection.")


def get_playlists(server: PlexServer, timeout: int) -> list[tuple[str, str, int, str]]:
    root = plex_xml(server, "/playlists", params={"playlistType": "audio"}, timeout=timeout)
    return [(p.attrib.get("ratingKey", ""), p.attrib.get("title", "Unnamed"), safe_int(p.attrib.get("leafCount")), human_duration(p.attrib.get("duration"))) for p in root.findall("Playlist") if p.attrib.get("ratingKey")]


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
            selected = [(playlists[i - 1][0], playlists[i - 1][1]) for i in default_indices]
            if default_random:
                selected.append(("Random", "Random"))
            return selected

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
                    continue
                idx = int(part)
                if not 1 <= idx <= len(playlists):
                    raise ValueError
                if idx not in indices:
                    indices.append(idx)

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
    )


def fetch_playlist_tracks(server: PlexServer, playlist_id: str, timeout: int) -> list[Track]:
    root = plex_xml(server, f"/playlists/{playlist_id}/items", timeout=timeout)
    return [t for item in root.findall("Track") if (t := track_from_xml(item, server, playlist_id)) is not None]


def fetch_library_tracks(server: PlexServer, library_key: str, timeout: int) -> list[Track]:
    root = plex_xml(server, f"/library/sections/{library_key}/all", params={"type": 10}, timeout=timeout)
    return [t for item in root.findall("Track") if (t := track_from_xml(item, server, "Random")) is not None]


def unique_tracks(tracks: Iterable[Track]) -> list[Track]:
    result, seen = [], set()
    for track in tracks:
        identity = track.rating_key or f"{track.media_url}|{track.title}|{track.artist}|{track.album}"
        if identity not in seen:
            seen.add(identity)
            result.append(track)
    return result


def choose_directory_limit() -> int:
    print("\nDirectory limit\n  - Maximum 255 audio files per directory.\n  - Enter -1 for unlimited.\n")
    while True:
        ans = input("  Maximum files per directory [255]: ").strip()
        if not ans:
            return 255
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
    container = (track.container or "").casefold()
    codec = (track.audio_codec or "").casefold()
    url_lower = track.media_url.casefold().split("?", 1)[0]
    
    if container in ("mp3", "mpeg") or codec == "mp3" or url_lower.endswith(".mp3"):
        return "mp3"
    if container in ("aac", "m4a", "mp4") or codec in ("aac", "mp4a") or url_lower.endswith((".aac", ".m4a", ".mp4")):
        return "aac"
    if container in ("flac",) or codec == "flac" or url_lower.endswith(".flac"):
        return "flac"
    if container in ("ogg", "oga") or codec in ("ogg", "vorbis", "opus") or url_lower.endswith((".ogg", ".opus")):
        return "ogg"
    return container or codec or "mp3"


def source_matches_output(track: Track, output_format: str) -> bool:
    output_format = output_format.lower()
    container, codec = (track.container or "").casefold(), (track.audio_codec or "").casefold()
    url_lower = track.media_url.casefold().split("?", 1)[0]
    
    if output_format == "mp3":
        return container in ("mp3", "mpeg") or codec == "mp3" or url_lower.endswith(".mp3")
    if output_format in ("aac", "m4a"):
        return container in ("aac", "m4a", "mp4") or codec in ("aac", "mp4a") or url_lower.endswith((".aac", ".m4a", ".mp4"))
    return container == output_format or codec == output_format


def is_format_supported(track: Track, conversion_formats: list[str]) -> bool:
    track_fmt = get_track_format(track)
    for spec in conversion_formats:
        fmt = get_format_from_spec(spec)
        if track_fmt == fmt or source_matches_output(track, fmt):
            return True
    return False


def build_output_path(root: Path, playlist_name: str, position: int, track: Track, directory_limit: int, conversion_formats: list[str]) -> Path:
    playlist_root = root / sanitize_filename(playlist_name, "Music")
    if directory_limit == -1:
        directory, directory_position = playlist_root, position
    else:
        directory = playlist_root / f"{((position - 1) // directory_limit) + 1:03d}"
        directory_position = ((position - 1) % directory_limit) + 1

    if is_format_supported(track, conversion_formats):
        fmt = get_track_format(track)
    else:
        fmt = get_format_from_spec(conversion_formats[0])

    return directory / track_filename(track, directory_position, fmt)


def conversion_spec(config: dict) -> tuple[str, str]:
    formats = config["audio"].get("conversion_formats", [])
    if not isinstance(formats, list) or not formats:
        raise RuntimeError("audio.conversion_formats must contain at least one format.")
    spec = str(formats[0]).strip()
    parts = spec.split(":", 1)
    output_format, quality = parts[0].strip().lower(), parts[1].strip() if len(parts) == 2 else ""
    if not re.fullmatch(r"[a-z0-9._+-]+", output_format):
        raise RuntimeError(f"Invalid conversion format: {spec}")
    return output_format, quality


def conversion_threads(config: dict) -> int:
    val = config["audio"].get("conversion_threads", "auto")
    if isinstance(val, str) and val.strip().lower() == "auto":
        return os.cpu_count() or 1
    try:
        workers = int(val)
    except (TypeError, ValueError) as exc:
        raise RuntimeError('audio.conversion_threads must be "auto" or a positive integer.') from exc
    if workers <= 0:
        raise RuntimeError("audio.conversion_threads must be positive.")
    return workers


def check_program(program: str) -> bool:
    try:
        return subprocess.run([program, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0
    except OSError:
        return False


def ensure_ffmpeg() -> None:
    if not check_program("ffmpeg") or not check_program("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe are required for conversion but were not found in PATH.")


def has_libfdk_aac() -> bool:
    global _HAS_FDK_AAC
    if _HAS_FDK_AAC is None:
        try:
            res = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            _HAS_FDK_AAC = "libfdk_aac" in res.stdout
        except OSError:
            _HAS_FDK_AAC = False
    return _HAS_FDK_AAC


def free_space(path: Path) -> int:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0


def get_disk_stats(path: Path) -> tuple[int, int]:
    """Return (free_space, reserve_1_percent)."""
    try:
        usage = shutil.disk_usage(path)
        return usage.free, usage.total // 100
    except OSError:
        return free_space(path), 0


def random_fill_space_available(output_root: Path) -> bool:
    free, reserve = get_disk_stats(output_root)
    return free > reserve


def track_identity(track: Track) -> str:
    return "|".join((sanitize_filename(track.artist, "").casefold(), sanitize_filename(track.album, "").casefold(), sanitize_filename(track.title, "").casefold()))


def exported_track_keys(output_root: Path) -> set[str]:
    keys = set()
    if output_root.exists():
        for path in output_root.rglob("*.*"):
            if path.suffix.lower() in (".mp3", ".m4a", ".aac", ".flac", ".ogg"):
                parts = path.stem.split(" - ", 3)
                if len(parts) == 4:
                    _, artist, album, title = parts
                    keys.add("|".join((artist.casefold().strip(), album.casefold().strip(), title.casefold().strip())))
    return keys


def select_random_tracks(tracks: list[Track], output_root: Path) -> list[Track]:
    existing = exported_track_keys(output_root)
    candidates = [t for t in tracks if track_identity(t) not in existing]
    random.SystemRandom().shuffle(candidates)
    return candidates


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
            if norm_q in ("VBR", "HIGH", "HQ", ""):
                cmd.extend(["-vbr", "5"])
            elif re.fullmatch(r"[0-9]+K", norm_q):
                cmd.extend(["-b:a", norm_q.lower()])
            else:
                cmd.extend(["-vbr", "5"])
        else:
            cmd.extend(["-c:a", "aac"])
            if re.fullmatch(r"[0-9]+K", norm_q):
                cmd.extend(["-b:a", norm_q.lower()])
            else:
                cmd.extend(["-b:a", "256k"])
    elif output_format == "mp3":
        cmd.extend(["-map", "0:v?", "-c:v", "copy", "-id3v2_version", "3", "-c:a", "libmp3lame"])
        if norm_q in ("VBR", "HIGH", "HQ", "V0", ""):
            cmd.extend(["-q:a", "0"])
        elif re.fullmatch(r"V[0-9]", norm_q):
            cmd.extend(["-q:a", norm_q[1:]])
        elif re.fullmatch(r"[0-9]+K", norm_q):
            cmd.extend(["-b:a", norm_q.lower()])
        else:
            cmd.extend(["-q:a", "0"])
    else:
        cmd.extend(["-c:a", output_format])
        if re.fullmatch(r"[0-9]+K", norm_q):
            cmd.extend(["-b:a", norm_q.lower()])
        elif quality:
            cmd.extend(["-q:a", quality])

    cmd.extend([str(output)])
    return cmd


def probe_duration(path: Path) -> float | None:
    try:
        res = subprocess.run(["ffprobe", "-hide_banner", "-loglevel", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60, check=False)
        if res.returncode == 0:
            dur = float(res.stdout.strip())
            return dur if dur > 0 else None
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def duration_matches(path: Path, expected_ms: int) -> tuple[bool, str]:
    if expected_ms <= 0:
        return True, ""
    actual = probe_duration(path)
    if actual is None:
        return False, "unable to read output duration"
    expected = expected_ms / 1000
    if abs(actual - expected) > DURATION_TOLERANCE_SECONDS:
        return False, f"duration mismatch: got {human_duration(int(actual * 1000))}, expected {human_duration(expected_ms)}"
    return True, ""


def direct_file_valid(path: Path, track: Track) -> tuple[bool, str]:
    try:
        size = path.stat().st_size
    except OSError:
        return False, "file is missing"
    if size < 1024:
        return False, "file is too small"
    if track.source_size > 0 and abs(size - track.source_size) > 512 * 1024:
        return False, f"size mismatch: got {human_size(size)}, expected {human_size(track.source_size)}"
    return True, ""


def copy_track(track: Track, destination: Path, token: str, timeout: int) -> tuple[bool, int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(destination.name + ".part")
    
    max_internal_retries = 5
    for attempt in range(max_internal_retries):
        existing_bytes = part_path.stat().st_size if part_path.exists() else 0
        if track.source_size > 0 and existing_bytes >= track.source_size:
            break

        headers = {}
        if token:
            headers["X-Plex-Token"] = token
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"

        request = urllib.request.Request(track.media_url, headers=headers)
        written = existing_bytes
        connection_dropped = False

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                mode = "ab" if existing_bytes > 0 and status == 206 else "wb"
                if mode == "wb":
                    written = 0
                    if existing_bytes > 0:
                        unlink_quiet(part_path)

                with part_path.open(mode) as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        written += len(chunk)

            if track.source_size > 0 and written < track.source_size - 512 * 1024:
                connection_dropped = True

        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                unlink_quiet(part_path)
                time.sleep(1.0)
                continue
            if attempt == max_internal_retries - 1:
                unlink_quiet(part_path)
                return False, 0, str(exc)
            time.sleep(1.0)
            continue
        except Exception as exc:
            if attempt == max_internal_retries - 1:
                unlink_quiet(part_path)
                return False, 0, str(exc)
            time.sleep(1.0)
            continue

        if connection_dropped:
            if attempt < max_internal_retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            else:
                unlink_quiet(part_path)
                return False, 0, f"incomplete transfer: got {written} bytes, expected {track.source_size}"

        if (track.source_size > 0 and abs(written - track.source_size) <= 512 * 1024) or (track.source_size == 0 and written >= 1024):
            break

    try:
        size = part_path.stat().st_size if part_path.exists() else 0
        if (track.source_size > 0 and abs(size - track.source_size) > 512 * 1024) or size < 1024:
            unlink_quiet(part_path)
            return False, 0, f"incomplete transfer: got {size} bytes, expected {track.source_size}"
        os.replace(part_path, destination)
        return True, size, ""
    except Exception as exc:
        unlink_quiet(part_path)
        return False, 0, str(exc)


def convert_track(track: Track, destination: Path, token: str, timeout: int, output_format: str, quality: str) -> tuple[bool, int, str]:
    ensure_ffmpeg()
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(destination.name + ".part")
    unlink_quiet(part_path)
    
    cmd = ffmpeg_command(track=track, output=part_path, token=token, output_format=output_format, quality=quality)
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=max(60, timeout) * 30, check=False)
    except subprocess.TimeoutExpired:
        unlink_quiet(part_path)
        return False, 0, "ffmpeg timed out"
    except OSError as exc:
        return False, 0, str(exc)

    if res.returncode != 0:
        unlink_quiet(part_path)
        return False, 0, (res.stderr.strip() or f"ffmpeg exited with code {res.returncode}")[-1600:]

    try:
        size = part_path.stat().st_size
    except OSError:
        size = 0

    if size < 1024:
        unlink_quiet(part_path)
        return False, 0, "ffmpeg produced an empty or incomplete file"

    valid, error = duration_matches(part_path, track.duration_ms)
    if not valid:
        unlink_quiet(part_path)
        return False, 0, error

    try:
        os.replace(part_path, destination)
        return True, size, ""
    except OSError as exc:
        unlink_quiet(part_path)
        return False, 0, str(exc)


def file_is_valid(path: Path, track: Track, conversion_formats: list[str]) -> tuple[bool, str]:
    if not path.is_file():
        return False, "file is missing"
    direct_copy = is_format_supported(track, conversion_formats)
    return direct_file_valid(path, track) if direct_copy else duration_matches(path, track.duration_ms)


def download_track(job: DownloadJob, server: PlexServer, retries: int, retry_delay: float, timeout: int, conversion_formats: list[str], output_format: str, quality: str) -> DownloadResult:
    valid, _ = file_is_valid(job.destination, job.track, conversion_formats)
    if valid:
        size = job.destination.stat().st_size if job.destination.exists() else 0
        return DownloadResult(job=job, success=True, skipped=True, bytes_written=size, elapsed=0.0, attempts=0)

    started = time.monotonic()
    direct_copy = is_format_supported(job.track, conversion_formats)
    last_error = "unknown error"

    for attempt in range(1, retries + 1):
        success, size, error = copy_track(job.track, job.destination, server.token, timeout) if direct_copy else convert_track(job.track, job.destination, server.token, timeout, output_format, quality)
        if success:
            return DownloadResult(job=job, success=True, skipped=False, bytes_written=size, elapsed=(time.monotonic() - started), attempts=attempt)

        last_error = error
        if attempt < retries:
            scale_factor = retry_delay / (2 ** (retries - 1)) if retries > 0 else retry_delay
            time.sleep(min(retry_delay, scale_factor * (2 ** (attempt - 1))) + (job.index % 7) * 0.15)

    return DownloadResult(job=job, success=False, skipped=False, bytes_written=0, elapsed=(time.monotonic() - started), attempts=retries, error=last_error)


def print_result(result: DownloadResult, output_root: Path) -> None:
    global _RANDOM_INITIAL_FREE, _RANDOM_TARGET_FILL
    status = "-" if result.skipped else ("✓" if result.success else "✗")
    rate = result.bytes_written / result.elapsed if result.elapsed > 0 else 0
    remaining = free_space(output_root)
    label = f"{result.job.track.artist} - {result.job.track.title}"
    
    if result.job.total:
        prefix = f"  [{result.job.index:>4}/{result.job.total:<4}] {status} "
    else:
        pct = min(100.0, ((_RANDOM_INITIAL_FREE - remaining) / _RANDOM_TARGET_FILL) * 100.0) if _RANDOM_TARGET_FILL > 0 else 100.0
        prefix = f"  [{pct:>8.2f}%] {status} "

    rate_str = "" if result.skipped else human_rate(rate)
    suffix = f" {human_size(result.bytes_written):>10} {rate_str:>12} {human_size(remaining):>10}"

    term_width = shutil.get_terminal_size(fallback=(80, 20)).columns
    avail_width = max(10, term_width - display_width(prefix) - display_width(suffix) - 2)
    
    compacted = compact(label, avail_width)
    padded_label = pad_right(compacted, avail_width)
    print(f"{prefix}{padded_label}{suffix}")

    if not result.success:
        print(f"       ! {compact(' '.join(result.error.split()), max(10, term_width - 9))}")


def download_jobs(jobs: list[DownloadJob], output_root: Path, server: PlexServer, retries: int, retry_delay: float, timeout: int, max_workers: int, conversion_formats: list[str], output_format: str, quality: str) -> tuple[int, int, int]:
    downloaded, failed, written, interrupted = 0, 0, 0, False
    adaptive = AdaptiveConcurrency(max_workers)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="plex")
    pending, futures = list(jobs), set()

    try:
        while pending or futures:
            if interrupted:
                break
            while pending and len(futures) < adaptive.workers:
                futures.add(executor.submit(download_track, pending.pop(0), server, retries, retry_delay, timeout, conversion_formats, output_format, quality))

            if not futures:
                break

            done, futures = concurrent.futures.wait(futures, timeout=0.2, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                try:
                    res: DownloadResult = future.result()
                    if res.success:
                        downloaded += 1
                        written += res.bytes_written
                        if not res.skipped:
                            adaptive.success() if res.attempts <= 1 else adaptive.failure()
                    else:
                        failed += 1
                        adaptive.failure()
                    print_result(res, output_root)
                except Exception as exc:
                    failed += 1
                    adaptive.failure()
                    print(f"  [Error] ✗ Worker exception: {exc}")
    except KeyboardInterrupt:
        interrupted = True
        print("\n\r  Aborting... Please wait.", end="", flush=True)
        for f in futures:
            f.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        print("\r  Interrupted and safely stopped.             ")
        raise
    else:
        executor.shutdown(wait=True)

    return downloaded, failed, written


def download_playlist(tracks: list[Track], playlist_name: str, output_root: Path, directory_limit: int, server: PlexServer, retries: int, retry_delay: float, timeout: int, workers: int, conversion_formats: list[str], output_format: str, quality: str) -> tuple[int, int, int]:
    if not tracks:
        return 0, 0, 0
    jobs = [DownloadJob(i, len(tracks), t, build_output_path(output_root, playlist_name, i, t, directory_limit, conversion_formats)) for i, t in enumerate(tracks, 1)]
    
    print(f"\n  {playlist_name}")
    print(f"  {'─' * min(72, display_width(playlist_name) + 2)}")
    print(f"  Tracks:     {len(jobs):,}")
    print(f"  Parallel:   {workers}")
    print(f"  Conversion: {output_format}" + (f":{quality}" if quality else "") + "\n")
    
    return download_jobs(jobs, output_root, server, retries, retry_delay, timeout, workers, conversion_formats, output_format, quality)


def download_random_fill(tracks: list[Track], output_root: Path, directory_limit: int, server: PlexServer, retries: int, retry_delay: float, timeout: int, workers: int, conversion_formats: list[str], output_format: str, quality: str) -> tuple[int, int, int]:
    global _RANDOM_INITIAL_FREE, _RANDOM_TARGET_FILL
    candidates = select_random_tracks(tracks, output_root)
    if not candidates:
        print("\n  Random: no new tracks remain.")
        return 0, 0, 0

    free, reserve = get_disk_stats(output_root)
    _RANDOM_INITIAL_FREE = free
    _RANDOM_TARGET_FILL = max(1, free - reserve)

    random_root = output_root / "Random"
    existing_count = sum(1 for _ in random_root.rglob("*.*") if _.suffix.lower() in (".mp3", ".m4a", ".aac")) if random_root.exists() else 0
    position, cursor, downloaded, failed, written = existing_count + 1, 0, 0, 0, 0

    print("\n  Random")
    print("  ──────")
    print(f"  Library tracks:   {len(tracks):,}")
    print(f"  New candidates:   {len(candidates):,}")
    print(f"  Parallel:         {workers}")
    print(f"  Conversion:       {output_format}" + (f":{quality}" if quality else ""))
    print("  Fill mode:        until drive capacity")
    print(f"  Safety reserve:   {human_size(reserve)} (1%)\n")

    while cursor < len(candidates) and random_fill_space_available(output_root):
        batch = candidates[cursor : cursor + (workers * 2)]
        cursor += len(batch)
        jobs = [DownloadJob(position + i, 0, t, build_output_path(output_root, "Random", position + i, t, directory_limit, conversion_formats)) for i, t in enumerate(batch)]
        position += len(batch)

        d, f, w = download_jobs(jobs, output_root, server, retries, retry_delay, timeout, workers, conversion_formats, output_format, quality)
        downloaded += d; failed += f; written += w
        if not random_fill_space_available(output_root):
            break

    print(f"\n  Random fill stopped with {human_size(free_space(output_root))} remaining.")
    if cursor >= len(candidates):
        print("  Random fill exhausted all available new library tracks.")
    return downloaded, failed, written


def main() -> int:
    print("\n" + "=" * 78)
    print("                              plexamp-usb")
    print("=" * 78 + "\n")
    print("  Export Plex music to a car-friendly USB filesystem.\n")

    try:
        config = load_config()
        server = prompt_server(config)
        timeout = safe_int(config["plex"].get("timeout"), 30)

        library_key, section_name = select_music_library(server, timeout)
        print(f"\nMusic library: {section_name}")

        playlists = get_playlists(server, timeout)
        output_root = Path(__file__).resolve().parent / (str(config["output"].get("directory", DOWNLOAD_DIR)).strip() or DOWNLOAD_DIR)
        output_root.mkdir(parents=True, exist_ok=True)

        selected = choose_playlists(playlists, output_root)
        directory_limit = safe_int(config["output"].get("directory_limit"), 255)
        if directory_limit == 0 or directory_limit < -1:
            directory_limit = choose_directory_limit()

        retries = max(1, safe_int(config["download"].get("retries"), 3))
        retry_delay = max(0.1, float(config["download"].get("retry_delay", 2.0)))
        workers = conversion_threads(config)
        
        conversion_formats = config["audio"].get("conversion_formats", ["aac:vbr", "mp3:vbr"])
        output_format, quality = conversion_spec(config)
        has_random = any(t.casefold().strip() == "random" for _, t in selected)

        print("\nDownload configuration")
        print(f"  Server:       {server.name}")
        print(f"  Address:      {server.base_url}")
        print(f"  Conversion:   {output_format}" + (f":{quality}" if quality else ""))
        print(f"  Parallel:     {workers} workers (adaptive)")
        print(f"  Retries:      {retries}")
        print("  Directory:    " + ("unlimited" if directory_limit == -1 else str(directory_limit)))
        print(f"  Destination:  {output_root}")
        if has_random:
            print("  Random fill:  enabled")
        print("\nCollecting music...\n")

        selected_tracks, random_selected, random_tracks = [], False, []
        for playlist_id, playlist_name in selected:
            if playlist_name.casefold().strip() == "random":
                random_selected = True
                continue
            compacted_name = compact(playlist_name, 60)
            padded_name = pad_right(compacted_name, 60)
            print(f"  {padded_name}", end=" ", flush=True)
            tracks = unique_tracks(fetch_playlist_tracks(server, playlist_id, timeout))
            selected_tracks.append((playlist_name, tracks))
            print(f"{len(tracks):>6,} tracks")

        if random_selected:
            compacted_name = compact("Random", 60)
            padded_name = pad_right(compacted_name, 60)
            print(f"  {padded_name}", end=" ", flush=True)
            random_tracks = unique_tracks(fetch_library_tracks(server, library_key, timeout))
            print(f"{len(random_tracks):>6,} tracks")
        print()

        grand_downloaded, grand_failed, grand_written = 0, 0, 0
        for playlist_name, tracks in selected_tracks:
            if tracks:
                d, f, w = download_playlist(tracks, playlist_name, output_root, directory_limit, server, retries, retry_delay, timeout, workers, conversion_formats, output_format, quality)
                grand_downloaded += d; grand_failed += f; grand_written += w

        if random_selected:
            d, f, w = download_random_fill(random_tracks, output_root, directory_limit, server, retries, retry_delay, timeout, workers, conversion_formats, output_format, quality)
            grand_downloaded += d; grand_failed += f; grand_written += w

        print("\n" + "─" * 78)
        print("  Music download complete")
        print("─" * 78)
        print(f"  Completed               {grand_downloaded + grand_failed:,}")
        print(f"  Downloaded              {grand_downloaded:,}")
        print(f"  Failed                  {grand_failed:,}")
        print(f"  Written                 {human_size(grand_written):>10}")
        print(f"  Space remaining         {human_size(free_space(output_root)):>10}")
        print(f"  Location                {output_root}\n")

        if grand_failed:
            print("  ! Some downloads failed.\n    Run again to retry them.\n")
            return 2
        print("  ✓ All downloads completed successfully.\n")
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
