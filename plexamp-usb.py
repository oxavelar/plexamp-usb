#!/usr/bin/env python3
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
from typing import Any, Iterable
import unicodedata


APP_NAME = "plexamp-usb"
CONFIG_JSON = "settings.json"
CONFIG_PATH = Path(__file__).resolve().parent / CONFIG_JSON
DOWNLOAD_DIR = "Downloads"

DURATION_TOLERANCE_SECONDS = 2.0
ADAPTIVE_SUCCESS_THRESHOLD = 8
ADAPTIVE_MIN_WORKERS = 1

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
    """Dynamically adjust worker concurrency based on throughput performance."""

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


def display_width(text: str) -> int:
    """Calculate terminal display column width using standard East Asian Width rules."""
    width = 0
    for char in str(text):
        code = ord(char)
        if (0xFE00 <= code <= 0xFE0F) or (0xE0100 <= code <= 0xE01EF) or code == 0x200D:
            continue
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return width


def pad_right(text: str, total_width: int) -> str:
    text_str = str(text)
    w = display_width(text_str)
    return text_str + " " * max(0, total_width - w)


def compact(text: str, width: int) -> str:
    text_str = str(text)
    if display_width(text_str) <= width:
        return text_str
    if width <= 1:
        return text_str[:width]

    current_w, result = 0, []
    for char in text_str:
        code = ord(char)
        char_w = 0 if ((0xFE00 <= code <= 0xFE0F) or (0xE0100 <= code <= 0xE01EF) or code == 0x200D) else (2 if unicodedata.east_asian_width(char) in ("W", "F") else 1)
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


def human_duration(milliseconds: Any) -> str:
    seconds = safe_int(milliseconds) // 1000
    if seconds <= 0:
        return "unknown"

    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"


def sanitize_filename(value: str, fallback: str = "Unknown", max_bytes: int = 255) -> str:
    value = unicodedata.normalize("NFC", str(value or "")).replace("\x00", "")
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]', "_", value)
    value = re.sub(r"\s+", " ", value).strip().rstrip(". ") or fallback

    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
    if value.upper() in reserved:
        value = f"_{value}"

    if len(value.encode("utf-8")) <= max_bytes:
        return value

    suffix = f"…{stable_hash(value, 6)}"
    while value and len((value + suffix).encode("utf-8")) > max_bytes:
        value = value[:-1]

    return value + suffix if value else suffix.encode("utf-8")[:max_bytes].decode("utf-8", "ignore")


def stable_hash(value: str, length: int = 8) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def format_extension(fmt: str) -> str:
    ext_map = {"aac": ".m4a", "m4a": ".m4a", "mp4": ".m4a", "ogg": ".ogg", "opus": ".ogg", "vorbis": ".ogg"}
    return ext_map.get(fmt.lower(), f".{fmt.lower()}")


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
    """Retrieve available user accounts from the Plex Media Server."""
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
    """Select or prompt for a Plex user account and update configuration."""
    plex = config.get("plex", {})
    timeout = safe_int(plex.get("timeout"), 30)
    configured_user = str(plex.get("user") or "").strip()

    if configured_user:
        print(f"  User:    {configured_user}")
        return configured_user

    print("  Discovering Plex users...")
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
    container = (track.container or "").casefold()
    codec = (track.audio_codec or "").casefold()
    url_lower = track.media_url.casefold().split("?", 1)[0]
    
    if container in ("mp3", "mpeg") or codec == "mp3" or url_lower.endswith(".mp3"):
        return "mp3"
    if container in ("aac", "m4a", "mp4") or codec in ("aac", "mp4a") or url_lower.endswith((".aac", ".m4a", ".mp4")):
        return "aac"
    if container == "flac" or codec == "flac" or url_lower.endswith(".flac"):
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
    return any(
        track_fmt == get_format_from_spec(spec) or source_matches_output(track, get_format_from_spec(spec))
        for spec in conversion_formats
    )


def build_output_path(root: Path, playlist_name: str, position: int, track: Track, directory_limit: int, conversion_formats: list[str]) -> Path:
    playlist_root = root / sanitize_filename(playlist_name, "Music")
    if directory_limit == -1:
        directory, directory_position = playlist_root, position
    else:
        directory = playlist_root / f"{((position - 1) // directory_limit) + 1:03d}"
        directory_position = ((position - 1) % directory_limit) + 1

    fmt = get_track_format(track) if is_format_supported(track, conversion_formats) else get_format_from_spec(conversion_formats[0])
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
        if workers > 0:
            return workers
    except (TypeError, ValueError):
        pass
    raise RuntimeError('audio.conversion_threads must be "auto" or a positive integer.')


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
    """Retrieve available free space and 1% safety reserve."""
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


def cleanup_playlist_leftovers(output_root: Path, playlist_name: str, tracks: list[Track], directory_limit: int, conversion_formats: list[str]) -> None:
    """Prune stale files in the playlist directory that no longer match the current track list."""
    playlist_root = output_root / sanitize_filename(playlist_name, "Music")
    if not playlist_root.exists():
        return

    expected_paths = {
        build_output_path(output_root, playlist_name, i, t, directory_limit, conversion_formats).resolve()
        for i, t in enumerate(tracks, 1)
    }

    for path in playlist_root.rglob("*.*"):
        if path.is_file() and path.suffix.lower() in (".mp3", ".m4a", ".aac", ".flac", ".ogg"):
            if path.resolve() not in expected_paths:
                unlink_quiet(path)

    for dirpath, _, _ in os.walk(playlist_root, topdown=False):
        d = Path(dirpath)
        if d != playlist_root and not any(d.iterdir()):
            with contextlib.suppress(OSError):
                d.rmdir()


def cleanup_random_fill_leftovers(output_root: Path, all_library_tracks: list[Track]) -> None:
    """Prune orphaned tracks in the Random directory that no longer exist in the Plex library."""
    random_root = output_root / "Random"
    if not random_root.exists():
        return

    valid_keys = {track_identity(t) for t in all_library_tracks}

    for path in random_root.rglob("*.*"):
        if path.is_file() and path.suffix.lower() in (".mp3", ".m4a", ".aac", ".flac", ".ogg"):
            parts = path.stem.split(" - ", 3)
            if len(parts) == 4:
                _, artist, album, title = parts
                key = "|".join((artist.casefold().strip(), album.casefold().strip(), title.casefold().strip()))
                if key not in valid_keys:
                    unlink_quiet(path)

    for dirpath, _, _ in os.walk(random_root, topdown=False):
        d = Path(dirpath)
        if d != random_root and not any(d.iterdir()):
            with contextlib.suppress(OSError):
                d.rmdir()


def select_random_tracks(tracks: list[Track], output_root: Path) -> list[Track]:
    """Prune library-orphaned files while retaining active candidates and previously downloaded tracks."""
    cleanup_random_fill_leftovers(output_root, tracks)

    existing_keys = set()
    random_root = output_root / "Random"
    if random_root.exists():
        for path in random_root.rglob("*.*"):
            if path.is_file() and path.suffix.lower() in (".mp3", ".m4a", ".aac", ".flac", ".ogg"):
                parts = path.stem.split(" - ", 3)
                if len(parts) == 4:
                    _, artist, album, title = parts
                    existing_keys.add("|".join((artist.casefold().strip(), album.casefold().strip(), title.casefold().strip())))

    candidates = [t for t in tracks if track_identity(t) not in existing_keys]
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
            cmd.extend(["-b:a", norm_q.lower()] if re.fullmatch(r"[0-9]+K", norm_q) else ["-vbr", "5"])
        else:
            bitrate = norm_q.lower() if re.fullmatch(r"[0-9]+K", norm_q) else "256k"
            cmd.extend(["-c:a", "aac", "-b:a", bitrate])
        cmd.extend(["-f", "mp4"])
    elif output_format == "mp3":
        cmd.extend(["-map", "0:v?", "-c:v", "copy", "-id3v2_version", "3", "-c:a", "libmp3lame"])
        cmd.extend(["-b:a", norm_q.lower()] if re.fullmatch(r"[0-9]+K", norm_q) else ["-q:a", "0"])
        cmd.extend(["-f", "mp3"])
    elif output_format == "flac":
        cmd.extend(["-c:a", "flac", "-f", "flac"])
    elif output_format in ("ogg", "opus", "vorbis"):
        codec = "libopus" if output_format == "opus" else "libvorbis"
        cmd.extend(["-c:a", codec, "-f", "ogg"])
    else:
        cmd.extend(["-c:a", "copy"])

    cmd.append(str(output))
    return cmd


def download_direct(job: DownloadJob, token: str, timeout: int = 30) -> int:
    temp_path = job.destination.with_suffix(job.destination.suffix + ".part")
    job.destination.parent.mkdir(parents=True, exist_ok=True)
    unlink_quiet(temp_path)

    headers = {"User-Agent": f"{APP_NAME}/1.0"}
    if token:
        headers["X-Plex-Token"] = token

    req = urllib.request.Request(job.track.media_url, headers=headers)
    bytes_written = 0
    with urllib.request.urlopen(req, timeout=timeout) as response, temp_path.open("wb") as handle:
        while chunk := response.read(64 * 1024):
            handle.write(chunk)
            bytes_written += len(chunk)

    if bytes_written == 0:
        unlink_quiet(temp_path)
        raise RuntimeError("Downloaded file is empty.")

    os.replace(temp_path, job.destination)
    return bytes_written


def convert_track(job: DownloadJob, token: str, output_format: str, quality: str) -> int:
    temp_path = job.destination.with_suffix(job.destination.suffix + ".part")
    job.destination.parent.mkdir(parents=True, exist_ok=True)
    unlink_quiet(temp_path)

    cmd = ffmpeg_command(job.track, temp_path, token, output_format, quality)
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False)

    if proc.returncode != 0 or not temp_path.exists() or temp_path.stat().st_size == 0:
        unlink_quiet(temp_path)
        err = proc.stderr.strip() if proc.stderr else "FFmpeg conversion failed"
        raise RuntimeError(err)

    bytes_written = temp_path.stat().st_size
    os.replace(temp_path, job.destination)
    return bytes_written


def download_track(
    job: DownloadJob,
    token: str,
    output_format: str,
    quality: str,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> DownloadResult:
    if job.destination.exists() and job.destination.stat().st_size > 0:
        return DownloadResult(job=job, success=True, skipped=True, bytes_written=0, elapsed=0.0, attempts=0)

    start_time = time.monotonic()
    last_error = ""

    for attempt in range(1, retries + 1):
        try:
            if source_matches_output(job.track, output_format):
                bytes_written = download_direct(job, token)
            else:
                bytes_written = convert_track(job, token, output_format, quality)

            elapsed = max(0.001, time.monotonic() - start_time)
            return DownloadResult(
                job=job,
                success=True,
                skipped=False,
                bytes_written=bytes_written,
                elapsed=elapsed,
                attempts=attempt,
            )
        except Exception as exc:
            last_error = str(exc)
            unlink_quiet(job.destination.with_suffix(job.destination.suffix + ".part"))
            if attempt < retries:
                time.sleep(retry_delay * attempt)

    elapsed = max(0.001, time.monotonic() - start_time)
    return DownloadResult(
        job=job,
        success=False,
        skipped=False,
        bytes_written=0,
        elapsed=elapsed,
        attempts=retries,
        error=last_error,
    )


def process_download_queue(
    jobs: list[DownloadJob],
    token: str,
    output_format: str,
    quality: str,
    max_workers: int,
    retries: int,
    retry_delay: float,
) -> list[DownloadResult]:
    if not jobs:
        return []

    adaptive = AdaptiveConcurrency(max_workers)
    results: list[DownloadResult] = []
    completed = 0
    total = len(jobs)
    start_time = time.monotonic()
    total_bytes = 0

    print(f"\nProcessing {total:,} track(s) using up to {max_workers} worker(s)...\n")

    def _execute_job(j: DownloadJob) -> DownloadResult:
        return download_track(j, token, output_format, quality, retries, retry_delay)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_execute_job, job): job for job in jobs}

        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)
            completed += 1
            if not res.skipped and res.success:
                total_bytes += res.bytes_written
                adaptive.success()
            elif not res.success:
                adaptive.failure()

            elapsed = time.monotonic() - start_time
            rate = total_bytes / elapsed if elapsed > 0 else 0
            status_line = (
                f" [{completed/total*100:5.1f}%] {completed}/{total} | "
                f"{human_size(total_bytes)} | {human_rate(rate)} | "
                f"Track: {compact(res.job.track.title, 25)}"
            )
            print(f"\r{pad_right(status_line, 80)}", end="", flush=True)

    print()
    return results


def main() -> None:
    print(f"=== {APP_NAME} ===")
    config = load_config()

    ensure_ffmpeg()

    print("\n--- Plex Connection ---")
    server = prompt_server(config, CONFIG_PATH)
    user = prompt_user(server, config, CONFIG_PATH)
    server = PlexServer(
        name=server.name,
        host=server.host,
        port=server.port,
        protocol=server.protocol,
        token=server.token,
        user=user,
    )

    timeout = safe_int(config["plex"].get("timeout"), 30)

    print("\n--- Music Library ---")
    library_key, library_title = select_music_library(server, timeout=timeout)
    print(f"  Selected: {library_title}")

    playlists = get_playlists(server, timeout=timeout)
    output_root = Path(config["output"].get("directory", DOWNLOAD_DIR)).resolve()
    selected_playlists = choose_playlists(playlists, output_root)

    directory_limit = safe_int(config["output"].get("directory_limit"), 255)
    if directory_limit not in (-1, 255):
        directory_limit = choose_directory_limit()

    output_format, quality = conversion_spec(config)
    conversion_formats = config["audio"].get("conversion_formats", ["aac:vbr"])
    max_workers = conversion_threads(config)
    retries = safe_int(config["download"].get("retries"), 3)
    retry_delay = safe_float(config["download"].get("retry_delay"), 2.0)

    all_jobs: list[DownloadJob] = []
    has_random = False

    for rk, title in selected_playlists:
        if rk == "Random":
            has_random = True
            continue

        print(f"\nFetching tracks for playlist: {title}...")
        tracks = fetch_playlist_tracks(server, rk, timeout=timeout)
        cleanup_playlist_leftovers(output_root, title, tracks, directory_limit, conversion_formats)

        for idx, track in enumerate(tracks, 1):
            dest = build_output_path(output_root, title, idx, track, directory_limit, conversion_formats)
            all_jobs.append(DownloadJob(index=idx, total=len(tracks), track=track, destination=dest))

    results = process_download_queue(
        all_jobs,
        token=server.token,
        output_format=output_format,
        quality=quality,
        max_workers=max_workers,
        retries=retries,
        retry_delay=retry_delay,
    )

    if has_random:
        print("\n--- Random Fill Mode ---")
        lib_tracks = fetch_library_tracks(server, library_key, timeout=timeout)
        random_candidates = select_random_tracks(lib_tracks, output_root)
        random_root = output_root / "Random"
        random_root.mkdir(parents=True, exist_ok=True)

        random_jobs: list[DownloadJob] = []
        for idx, track in enumerate(random_candidates, 1):
            if not random_fill_space_available(output_root):
                print("  Disk space safety limit reached for Random Fill.")
                break
            dest = random_root / track_filename(track, idx, output_format)
            random_jobs.append(DownloadJob(index=idx, total=len(random_candidates), track=track, destination=dest))

        if random_jobs:
            random_results = process_download_queue(
                random_jobs,
                token=server.token,
                output_format=output_format,
                quality=quality,
                max_workers=max_workers,
                retries=retries,
                retry_delay=retry_delay,
            )
            results.extend(random_results)

    successes = [r for r in results if r.success and not r.skipped]
    skipped = [r for r in results if r.skipped]
    failures = [r for r in results if not r.success]

    total_bytes = sum(r.bytes_written for r in successes)
    total_time = sum(r.elapsed for r in results)

    print("\n--- Summary ---")
    print(f"  Downloaded: {len(successes):,} tracks ({human_size(total_bytes)})")
    print(f"  Skipped:    {len(skipped):,} tracks (already present)")
    print(f"  Failed:     {len(failures):,} tracks")
    if total_time > 0:
        print(f"  Total time: {human_duration(int(total_time * 1000))}")
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
