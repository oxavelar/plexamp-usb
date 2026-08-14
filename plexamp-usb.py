#!/usr/bin/env python3
"""Export Plex music to a car-friendly USB filesystem.

The program discovers a reachable Plex Media Server, authenticates only when
local unauthenticated access is unavailable, lets the user select Plex music
playlists, copies already-supported audio formats directly, converts other
formats according to the configured conversion rule, and writes everything
below a Downloads directory beside this script.

Downloads are deterministic and resumable: completed files are retained,
invalid or incomplete files are replaced, and subsequent runs only transfer
what is missing or invalid.

Directories can be capped at a user-selected number of audio files to
accommodate car stereo filesystem limits. A value of -1 disables the limit.

A playlist named "Random" is treated as a special fill mode. When selected,
tracks are sampled from the complete Plex music library and downloaded until
the destination filesystem reaches its available-space safety threshold.
Existing exported tracks are skipped so repeated runs continue filling the
drive with new music.

Configuration is read from settings.json beside this script. Interactive
settings are written back so they become the defaults for the next run.

Requirements:
    - Python 3.10+
    - ffmpeg only when conversion is required
"""

from __future__ import annotations

import concurrent.futures
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
import unicodedata

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


APP_NAME = "plexamp-usb"
CONFIG_JSON = "settings.json"
DOWNLOAD_DIR = "Downloads"

RANDOM_FILL_RESERVE = 128 * 1024 * 1024

DEFAULT_CONFIG = {
    "plex": {
        "host": "",
        "port": 32400,
        "token": "",
        "timeout": 30,
    },
    "output": {
        "directory": DOWNLOAD_DIR,
    },
    "audio": {
        "supported_formats": ["mp3"],
        "conversion": "mp3:V0",
    },
    "download": {
        "retries": 5,
        "retry_delay": 2.0,
        "directory_limit": 255,
    },
}


@dataclass(frozen=True)
class PlexServer:
    """Connection information for a Plex Media Server."""

    name: str
    host: str
    port: int = 32400
    protocol: str = "http"
    token: str = ""

    @property
    def base_url(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}"


@dataclass(frozen=True)
class Conversion:
    """Parsed audio conversion rule."""

    format: str
    quality: str


@dataclass(frozen=True)
class Track:
    """Metadata and source information for one Plex audio track."""

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
    """One deterministic output location for a Plex track."""

    index: int
    total: int
    track: Track
    destination: Path
    output_format: str


@dataclass(frozen=True)
class DownloadResult:
    """Result of one download attempt sequence."""

    job: DownloadJob
    success: bool
    skipped: bool
    bytes_written: int
    elapsed: float
    attempts: int
    error: str = ""


def compact(text: str, width: int) -> str:
    """Fit text into a fixed console width without breaking the layout."""
    text = str(text)

    if len(text) <= width:
        return text

    return text[: max(0, width - 1)] + "…" if width > 1 else text[:width]


def human_size(value: int | float) -> str:
    """Format a byte count using binary units."""
    if value < 1024:
        return f"{value:.1f} B"

    size = float(value)

    for unit in ("KB", "MB", "GB", "TB", "PB"):
        size /= 1024

        if size < 1024:
            return f"{size:.1f} {unit}"

    return f"{size:.1f} EB"


def human_rate(value: float) -> str:
    """Format a transfer rate."""
    return f"{human_size(value)}/s"


def human_duration(milliseconds: int | str | None) -> str:
    """Format a Plex duration in milliseconds."""
    try:
        seconds = int(float(milliseconds or 0)) // 1000
    except (TypeError, ValueError):
        return "unknown"

    if seconds <= 0:
        return "unknown"

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m"

    if minutes:
        return f"{minutes}m {seconds}s"

    return f"{seconds}s"


def stable_hash(value: str, length: int = 8) -> str:
    """Return a short deterministic hash."""
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def sanitize_filename(
    value: str,
    fallback: str = "Unknown",
    max_bytes: int = 255,
) -> str:
    """Return a deterministic filename-safe Unicode component."""
    value = unicodedata.normalize("NFC", str(value or ""))
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]', "_", value)
    value = re.sub(r"\s+", " ", value).strip().rstrip(". ")

    if not value:
        value = fallback

    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }

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

    return suffix.encode("utf-8")[:max_bytes].decode(
        "utf-8",
        "ignore",
    )


def track_filename(
    track: Track,
    number: int,
    extension: str,
) -> str:
    """Build a stable car-friendly filename."""
    artist = sanitize_filename(track.artist, "Unknown Artist")
    album = sanitize_filename(track.album, "Unknown Album")
    title = sanitize_filename(track.title, "Unknown Track")

    try:
        track_number = int(track.index)
    except (TypeError, ValueError):
        track_number = 0

    prefix = f"{track_number:02d}" if track_number else f"{number:02d}"
    ext = f".{extension}"

    prefix_part = f"{prefix} - {artist} - {album} - "
    available = 255 - len((prefix_part + ext).encode("utf-8"))

    if available <= 0:
        album = sanitize_filename(
            album,
            "Unknown Album",
            max_bytes=80,
        )
        prefix_part = f"{prefix} - {artist} - {album} - "
        available = 255 - len((prefix_part + ext).encode("utf-8"))

    title = sanitize_filename(
        title,
        "Unknown Track",
        max_bytes=max(1, available),
    )

    return f"{prefix_part}{title}{ext}"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge configuration overrides into defaults."""
    result = dict(base)

    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def config_path() -> Path:
    """Return the configuration path beside this script."""
    return Path(__file__).resolve().parent / CONFIG_JSON


def save_config(config: dict) -> None:
    """Persist current settings for the next run."""
    path = config_path()

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            config,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")


def load_config() -> dict:
    """Load settings, creating defaults when necessary."""
    path = config_path()

    if not path.exists():
        config = deep_merge({}, DEFAULT_CONFIG)
        save_config(config)
        print(f"  Created {path}")
        return config

    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON in {path}: {exc}"
        ) from exc

    if not isinstance(loaded, dict):
        raise RuntimeError(f"Invalid configuration: {path}")

    return deep_merge(DEFAULT_CONFIG, loaded)


def parse_conversion(value: str) -> Conversion:
    """Parse a conversion rule such as ``mp3:V0``."""
    value = str(value or "").strip()

    if ":" not in value:
        raise RuntimeError(
            "audio.conversion must use format:quality, "
            "for example mp3:V0."
        )

    output_format, quality = (
        part.strip().lower()
        for part in value.split(":", 1)
    )

    if not output_format or not quality:
        raise RuntimeError(
            "audio.conversion must use format:quality, "
            "for example mp3:V0."
        )

    return Conversion(
        format=output_format,
        quality=quality.upper(),
    )


def validate_audio_config(config: dict) -> Conversion:
    """Validate audio settings without restricting conversion targets."""
    audio = config["audio"]

    formats = audio.get("supported_formats", [])

    if not isinstance(formats, list) or not formats:
        raise RuntimeError(
            "audio.supported_formats must be a non-empty list."
        )

    normalized = [
        str(value).strip().lower()
        for value in formats
        if str(value).strip()
    ]

    if not normalized:
        raise RuntimeError(
            "audio.supported_formats must contain at least one format."
        )

    audio["supported_formats"] = list(dict.fromkeys(normalized))

    # The conversion target may intentionally also be supported. That is what
    # makes a configured format copyable when already present at the source.
    conversion = parse_conversion(
        audio.get("conversion", "mp3:V0")
    )

    audio["conversion"] = (
        f"{conversion.format}:{conversion.quality}"
    )

    return conversion


def build_request(
    url: str,
    token: str = "",
    accept: str = "*/*",
) -> urllib.request.Request:
    """Create a Plex HTTP request."""
    headers = {
        "User-Agent": f"{APP_NAME}/1.0",
        "Accept": accept,
    }

    if token:
        headers["X-Plex-Token"] = token

    return urllib.request.Request(
        url,
        headers=headers,
    )


def http_get(
    url: str,
    token: str = "",
    timeout: int = 30,
) -> bytes:
    """Fetch a complete HTTP response."""
    request = build_request(
        url,
        token=token,
        accept="application/xml,application/json,text/plain,*/*",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def plex_xml(
    server: PlexServer,
    path: str,
    params: dict | None = None,
    timeout: int = 30,
) -> ET.Element:
    """Request and parse one Plex XML endpoint."""
    query = urllib.parse.urlencode(params or {})
    url = f"{server.base_url}{path}"

    if query:
        url += f"?{query}"

    try:
        payload = http_get(
            url,
            token=server.token,
            timeout=timeout,
        )
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Plex returned HTTP {exc.code} for {path}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Unable to reach Plex: {exc.reason}"
        ) from exc

    try:
        return ET.fromstring(payload)
    except ET.ParseError as exc:
        raise RuntimeError(
            f"Plex returned invalid XML for {path}"
        ) from exc


def test_server(
    server: PlexServer,
    timeout: int,
) -> ET.Element | None:
    """Test Plex connectivity and return its identity."""
    try:
        return plex_xml(
            server,
            "/identity",
            timeout=timeout,
        )
    except Exception:
        return None


def discover_gdm_servers(
    timeout: float = 3.0,
) -> list[PlexServer]:
    """Discover Plex Media Servers using Plex GDM."""
    discovered: dict[tuple[str, int], PlexServer] = {}

    message = (
        b"M-SEARCH * HTTP/1.0\r\n"
        b"HOST: 239.0.0.250:32414\r\n"
        b'MAN: "ssdp:discover"\r\n'
        b"ST: plex/media-server\r\n"
        b"\r\n"
    )

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
        socket.IPPROTO_UDP,
    )
    sock.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_BROADCAST,
        1,
    )
    sock.settimeout(0.5)

    try:
        sock.sendto(
            message,
            ("239.0.0.250", 32414),
        )

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            try:
                data, address = sock.recvfrom(8192)
            except socket.timeout:
                continue
            except OSError:
                break

            lines = data.decode(
                "utf-8",
                errors="replace",
            ).splitlines()

            if not lines:
                continue

            headers: dict[str, str] = {}

            for line in lines[1:]:
                if ":" not in line:
                    continue

                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()

            if "plex/media-server" not in headers.get(
                "content-type",
                "",
            ).lower():
                continue

            host = headers.get("host", "").strip() or address[0]

            if host.endswith(".plex.direct"):
                host = address[0]

            try:
                port = int(headers.get("port", "32400"))
            except ValueError:
                port = 32400

            name = headers.get("name", host)

            discovered[(host, port)] = PlexServer(
                name=name,
                host=host,
                port=port,
            )
    finally:
        sock.close()

    return list(discovered.values())


def prompt_server(config: dict) -> PlexServer:
    """Select a Plex server, preferring local unauthenticated access."""
    plex = config["plex"]

    timeout = int(plex.get("timeout", 30))
    configured_host = str(plex.get("host") or "").strip()
    configured_port = int(plex.get("port") or 32400)
    configured_token = str(plex.get("token") or "").strip()

    if configured_host:
        configured = PlexServer(
            name=configured_host,
            host=configured_host,
            port=configured_port,
            token=configured_token,
        )

        identity = test_server(configured, timeout)

        if identity is not None:
            print(
                f"  Server: {identity.attrib.get('friendlyName', configured_host)}"
            )
            print(f"  Address: {configured.base_url}")
            print("  Access:  OK")

            return PlexServer(
                name=identity.attrib.get(
                    "friendlyName",
                    configured_host,
                ),
                host=configured_host,
                port=configured_port,
                token=configured_token,
            )

        print(
            f"  Configured server unavailable: {configured.base_url}"
        )

    print("  Discovering local Plex servers...")

    servers = discover_gdm_servers()

    if not servers:
        print()
        print("  No local Plex server was discovered.")
        print()
        print("  Enter the Plex server address manually,")
        print("  or set plex.host in settings.json.")
        print()

        host = input("  Plex server address: ").strip()

        if not host:
            raise RuntimeError(
                "No Plex server address supplied."
            )

        parsed = urllib.parse.urlparse(host)

        if parsed.scheme:
            protocol = parsed.scheme
            host = parsed.hostname or host
            port = parsed.port or configured_port
        else:
            protocol = "http"

            if ":" in host:
                possible_host, possible_port = host.rsplit(":", 1)

                try:
                    port = int(possible_port)
                    host = possible_host
                except ValueError:
                    port = configured_port
            else:
                port = configured_port

        servers = [
            PlexServer(
                name=host,
                host=host,
                port=port,
                protocol=protocol,
            )
        ]

    if len(servers) == 1:
        server = servers[0]
    else:
        print()
        print("  Plex servers:")

        for index, candidate in enumerate(servers, 1):
            print(
                f"    {index}. {candidate.name} "
                f"({candidate.base_url})"
            )

        while True:
            answer = input(
                "\n  Select Plex server [1]: "
            ).strip()

            if not answer:
                server = servers[0]
                break

            try:
                index = int(answer)

                if 1 <= index <= len(servers):
                    server = servers[index - 1]
                    break
            except ValueError:
                pass

            print("  Invalid selection.")

    print(
        f"  Found: {server.name} ({server.base_url})"
    )

    identity = test_server(server, timeout)

    if identity is not None:
        print("  Local access: OK")

        return PlexServer(
            name=identity.attrib.get(
                "friendlyName",
                server.name,
            ),
            host=server.host,
            port=server.port,
            protocol=server.protocol,
        )

    print()
    print("  Local access requires authentication.")

    token = configured_token

    if not token:
        token = input("  Plex token: ").strip()

    if not token:
        raise RuntimeError(
            "A Plex token is required for this server."
        )

    authenticated = PlexServer(
        name=server.name,
        host=server.host,
        port=server.port,
        protocol=server.protocol,
        token=token,
    )

    identity = test_server(authenticated, timeout)

    if identity is None:
        raise RuntimeError(
            "Unable to connect to Plex with the supplied token."
        )

    print("  Authentication: OK")

    return PlexServer(
        name=identity.attrib.get(
            "friendlyName",
            server.name,
        ),
        host=server.host,
        port=server.port,
        protocol=server.protocol,
        token=token,
    )


def select_music_library(
    server: PlexServer,
    timeout: int,
) -> tuple[str, str]:
    """Select a Plex music library."""
    root = plex_xml(
        server,
        "/library/sections",
        timeout=timeout,
    )

    libraries = []

    for directory in root.findall("Directory"):
        if directory.attrib.get("type") != "artist":
            continue

        key = directory.attrib.get("key", "")
        title = directory.attrib.get("title", "Music")

        if key:
            libraries.append((key, title))

    if not libraries:
        raise RuntimeError(
            "No Plex music library was found."
        )

    if len(libraries) == 1:
        return libraries[0]

    print()
    print("Music libraries:")
    print()

    for index, (_, title) in enumerate(libraries, 1):
        print(f"  {index:>2}. {title}")

    while True:
        answer = input(
            "\nSelect music library [1]: "
        ).strip()

        if not answer:
            return libraries[0]

        try:
            index = int(answer)

            if 1 <= index <= len(libraries):
                return libraries[index - 1]
        except ValueError:
            pass

        print("Invalid selection.")


def get_playlists(
    server: PlexServer,
    timeout: int,
) -> list[tuple[str, str, int, str]]:
    """Return audio playlists available from Plex."""
    root = plex_xml(
        server,
        "/playlists",
        params={"playlistType": "audio"},
        timeout=timeout,
    )

    playlists = []

    for playlist in root.findall("Playlist"):
        rating_key = playlist.attrib.get("ratingKey", "")
        title = playlist.attrib.get("title", "Unnamed")

        if not rating_key:
            continue

        try:
            count = int(playlist.attrib.get("leafCount", 0))
        except ValueError:
            count = 0

        playlists.append(
            (
                rating_key,
                title,
                count,
                human_duration(
                    playlist.attrib.get("duration")
                ),
            )
        )

    return playlists


def choose_playlists(
    playlists: list[tuple[str, str, int, str]],
) -> list[tuple[str, str]]:
    """Prompt for one or more music playlist downloads."""
    if not playlists:
        raise RuntimeError(
            "No Plex music playlists were found."
        )

    print()
    print("Music playlists:")
    print()

    for index, (_, title, count, duration) in enumerate(
        playlists,
        1,
    ):
        print(
            f"  {index:>2}. {title} "
            f"({count:,} tracks, {duration})"
        )

    print("   X. Random Fill Mode")
    print("   A. All")
    print("   Multiple selections: 1,3,5,X")

    while True:
        answer = input(
            "\nSelect downloads: "
        ).strip()

        if answer.lower() == "a":
            return [
                (rating_key, title)
                for rating_key, title, _, _ in playlists
            ]

        try:
            indices = []
            random_selected = False

            for part in answer.split(","):
                part = part.strip()

                if part.lower() == "x":
                    random_selected = True
                    continue

                index = int(part)

                if not 1 <= index <= len(playlists):
                    raise ValueError

                if index not in indices:
                    indices.append(index)

            selected = [
                (
                    playlists[index - 1][0],
                    playlists[index - 1][1],
                )
                for index in indices
            ]

            if random_selected:
                selected.append(("Random", "Random"))

            if not selected:
                raise ValueError

            return selected

        except ValueError:
            print("Invalid selection.")


def make_track(
    server: PlexServer,
    item: ET.Element,
    playlist_id: str,
) -> Track | None:
    """Build a Track from a Plex track element."""
    media = item.find("Media")

    if media is None:
        return None

    part = media.find("Part")

    if part is None:
        return None

    key = part.attrib.get("key", "")

    if not key:
        return None

    try:
        source_size = int(part.attrib.get("size", 0))
    except ValueError:
        source_size = 0

    try:
        duration_ms = int(item.attrib.get("duration", 0))
    except ValueError:
        duration_ms = 0

    return Track(
        rating_key=item.attrib.get("ratingKey", ""),
        title=item.attrib.get("title", "Unknown Track"),
        artist=item.attrib.get(
            "grandparentTitle",
            item.attrib.get("originalTitle", "Unknown Artist"),
        ),
        album=item.attrib.get(
            "parentTitle",
            "Unknown Album",
        ),
        album_artist=item.attrib.get(
            "parentTitle",
            "",
        ),
        parent_index=item.attrib.get(
            "parentIndex",
            "1",
        ),
        index=item.attrib.get("index", "0"),
        duration_ms=duration_ms,
        media_url=urllib.parse.urljoin(
            server.base_url,
            key,
        ),
        source_size=source_size,
        playlist_id=playlist_id,
        container=part.attrib.get("container", ""),
        audio_codec=part.attrib.get("audioCodec", ""),
    )


def fetch_playlist_tracks(
    server: PlexServer,
    playlist_id: str,
    timeout: int,
) -> list[Track]:
    """Fetch playable audio tracks from one Plex playlist."""
    root = plex_xml(
        server,
        f"/playlists/{playlist_id}/items",
        timeout=timeout,
    )

    tracks = []

    for item in root:
        track = make_track(
            server,
            item,
            playlist_id,
        )

        if track is not None:
            tracks.append(track)

    return tracks


def fetch_library_tracks(
    server: PlexServer,
    library_key: str,
    timeout: int,
) -> list[Track]:
    """Fetch all playable audio tracks from a Plex music library."""
    root = plex_xml(
        server,
        f"/library/sections/{library_key}/all",
        params={"type": 10},
        timeout=timeout,
    )

    tracks = []

    for item in root.findall("Track"):
        track = make_track(
            server,
            item,
            "Random",
        )

        if track is not None:
            tracks.append(track)

    return tracks


def unique_tracks(
    tracks: Iterable[Track],
) -> list[Track]:
    """Remove duplicate playlist entries while preserving order."""
    result = []
    seen = set()

    for track in tracks:
        identity = (
            track.rating_key
            or f"{track.media_url}|{track.title}|"
            f"{track.artist}|{track.album}"
        )

        if identity in seen:
            continue

        seen.add(identity)
        result.append(track)

    return result


def choose_directory_limit(config: dict) -> int:
    """Prompt for the maximum files per directory."""
    current = int(
        config["download"].get(
            "directory_limit",
            255,
        )
    )

    print()
    print("Directory limit")
    print()
    print("  - Maximum 255 audio files per directory.")
    print("  - Enter -1 for unlimited.")
    print()

    while True:
        answer = input(
            f"  Maximum files per directory [{current}]: "
        ).strip()

        if not answer:
            return current

        try:
            value = int(answer)

            if value == -1 or value > 0:
                return value
        except ValueError:
            pass

        print(
            "  Enter a positive number or -1."
        )


def playlist_directory(name: str) -> str:
    """Return a safe, stable playlist directory name."""
    return sanitize_filename(name, "Music")


def build_output_path(
    root: Path,
    playlist_name: str,
    position: int,
    track: Track,
    directory_limit: int,
    extension: str,
) -> Path:
    """Return the deterministic destination for one track."""
    playlist_root = root / playlist_directory(playlist_name)

    if directory_limit == -1:
        directory = playlist_root
        directory_position = position
    else:
        directory_number = ((position - 1) // directory_limit) + 1
        directory_position = ((position - 1) % directory_limit) + 1
        directory = playlist_root / f"{directory_number:03d}"

    return directory / track_filename(
        track,
        directory_position,
        extension,
    )


def file_is_valid(path: Path) -> bool:
    """Check that an exported file exists and has usable content."""
    try:
        return path.is_file() and path.stat().st_size >= 1024
    except OSError:
        return False


def check_ffmpeg() -> None:
    """Verify FFmpeg only when conversion will be needed."""
    try:
        completed = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(
            "ffmpeg is required for conversion but was not found in PATH."
        ) from exc

    if completed.returncode != 0:
        raise RuntimeError(
            "ffmpeg could not be executed."
        )


def determine_workers() -> int:
    """Choose bounded parallelism for Plex and USB I/O."""
    cpu_count = os.cpu_count() or 2
    return max(2, min(8, cpu_count))


def free_space(path: Path) -> int:
    """Return free bytes at a filesystem location."""
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0


def random_fill_space_available(
    output_root: Path,
) -> bool:
    """Return whether random fill can safely continue."""
    return free_space(output_root) > RANDOM_FILL_RESERVE


def exported_track_keys(
    output_root: Path,
) -> set[str]:
    """Build normalized identities from exported filenames."""
    keys = set()

    if not output_root.exists():
        return keys

    for path in output_root.rglob("*"):
        if not path.is_file():
            continue

        stem = path.stem
        parts = stem.split(" - ", 3)

        if len(parts) != 4:
            continue

        _, artist, album, title = parts

        keys.add(
            "|".join(
                (
                    artist.casefold().strip(),
                    album.casefold().strip(),
                    title.casefold().strip(),
                )
            )
        )

    return keys


def track_identity(track: Track) -> str:
    """Return a normalized identity for duplicate detection."""
    return "|".join(
        (
            sanitize_filename(track.artist, "").casefold(),
            sanitize_filename(track.album, "").casefold(),
            sanitize_filename(track.title, "").casefold(),
        )
    )


def select_random_tracks(
    tracks: list[Track],
    output_root: Path,
) -> list[Track]:
    """Randomize the library and remove tracks already exported."""
    existing = exported_track_keys(output_root)

    candidates = [
        track
        for track in tracks
        if track_identity(track) not in existing
    ]

    random.SystemRandom().shuffle(candidates)
    return candidates


def source_format(track: Track) -> str:
    """Return the best Plex-reported source container."""
    container = (track.container or "").strip().lower()

    if container:
        return container

    codec = (track.audio_codec or "").strip().lower()

    if codec:
        return codec

    path = urllib.parse.urlparse(track.media_url).path
    extension = Path(path).suffix.lower().lstrip(".")

    return extension


def direct_copy_allowed(
    track: Track,
    supported_formats: set[str],
) -> bool:
    """Return whether the source can be copied without conversion."""
    return source_format(track) in supported_formats


def ffmpeg_quality(quality: str) -> str:
    """Convert friendly quality names into FFmpeg's expected value."""
    quality = quality.strip().upper()

    presets = {
        "V0": "0",
        "V1": "1",
        "V2": "2",
        "V3": "3",
        "V4": "4",
        "V5": "5",
        "V6": "6",
        "V7": "7",
        "V8": "8",
        "V9": "9",
    }

    return presets.get(quality, quality)


def ffmpeg_command(
    url: str,
    output: Path,
    token: str,
    conversion: Conversion,
) -> list[str]:
    """Build an FFmpeg command for the configured conversion."""
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_on_network_error",
        "1",
        "-reconnect_on_http_error",
        "429,500,502,503,504",
        "-reconnect_delay_max",
        "5",
    ]

    if token:
        command.extend(
            [
                "-headers",
                f"X-Plex-Token: {token}\r\n",
            ]
        )

    command.extend(
        [
            "-i",
            url,
            "-vn",
            "-map",
            "0:a:0",
            "-map_metadata",
            "0",
        ]
    )

    if conversion.format == "mp3":
        command.extend(
            [
                "-c:a",
                "libmp3lame",
                "-q:a",
                ffmpeg_quality(conversion.quality),
                "-f",
                "mp3",
            ]
        )
    else:
        raise RuntimeError(
            f"Unsupported conversion format: {conversion.format}"
        )

    command.append(str(output))
    return command


def direct_download(
    url: str,
    destination: Path,
    token: str,
    timeout: int,
) -> tuple[bool, int, str]:
    """Stream an already-supported source directly to disk."""
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    part_path = destination.with_name(
        destination.name + ".part"
    )

    try:
        part_path.unlink(missing_ok=True)
    except OSError:
        pass

    try:
        request = build_request(
            url,
            token=token,
        )

        with urllib.request.urlopen(
            request,
            timeout=max(60, timeout),
        ) as response:
            with part_path.open("wb") as output:
                shutil.copyfileobj(
                    response,
                    output,
                    length=1024 * 1024,
                )

    except Exception as exc:
        try:
            part_path.unlink(missing_ok=True)
        except OSError:
            pass

        return False, 0, str(exc)

    try:
        size = part_path.stat().st_size
    except OSError:
        size = 0

    if size < 1024:
        try:
            part_path.unlink(missing_ok=True)
        except OSError:
            pass

        return False, 0, "download produced an empty or incomplete file"

    try:
        os.replace(part_path, destination)
    except OSError as exc:
        try:
            part_path.unlink(missing_ok=True)
        except OSError:
            pass

        return False, 0, str(exc)

    return True, size, ""


def run_ffmpeg(
    url: str,
    destination: Path,
    token: str,
    timeout: int,
    conversion: Conversion,
) -> tuple[bool, int, str]:
    """Convert one track into an atomic temporary output file."""
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    part_path = destination.with_name(
        destination.name + ".part"
    )

    try:
        part_path.unlink(missing_ok=True)
    except OSError:
        pass

    command = ffmpeg_command(
        url=url,
        output=part_path,
        token=token,
        conversion=conversion,
    )

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(60, timeout) * 30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        try:
            part_path.unlink(missing_ok=True)
        except OSError:
            pass

        return False, 0, "ffmpeg timed out"
    except OSError as exc:
        return False, 0, str(exc)

    if completed.returncode != 0:
        error = (
            completed.stderr.strip()
            or f"ffmpeg exited with code {completed.returncode}"
        )

        try:
            part_path.unlink(missing_ok=True)
        except OSError:
            pass

        return False, 0, error[-1600:]

    try:
        size = part_path.stat().st_size
    except OSError:
        size = 0

    if size < 1024:
        try:
            part_path.unlink(missing_ok=True)
        except OSError:
            pass

        return False, 0, "ffmpeg produced an empty or incomplete file"

    try:
        os.replace(part_path, destination)
    except OSError as exc:
        try:
            part_path.unlink(missing_ok=True)
        except OSError:
            pass

        return False, 0, str(exc)

    return True, size, ""


def download_track(
    job: DownloadJob,
    server: PlexServer,
    supported_formats: set[str],
    conversion: Conversion,
    retries: int,
    retry_delay: float,
    timeout: int,
) -> DownloadResult:
    """Copy or convert one track with retry-safe semantics."""
    if file_is_valid(job.destination):
        try:
            size = job.destination.stat().st_size
        except OSError:
            size = 0

        return DownloadResult(
            job=job,
            success=True,
            skipped=True,
            bytes_written=size,
            elapsed=0.0,
            attempts=0,
        )

    started = time.monotonic()
    last_error = "unknown error"
    direct = direct_copy_allowed(
        job.track,
        supported_formats,
    )

    for attempt in range(1, retries + 1):
        if direct:
            success, size, error = direct_download(
                url=job.track.media_url,
                destination=job.destination,
                token=server.token,
                timeout=timeout,
            )
        else:
            success, size, error = run_ffmpeg(
                url=job.track.media_url,
                destination=job.destination,
                token=server.token,
                timeout=timeout,
                conversion=conversion,
            )

        if success:
            return DownloadResult(
                job=job,
                success=True,
                skipped=False,
                bytes_written=size,
                elapsed=time.monotonic() - started,
                attempts=attempt,
            )

        last_error = error

        if attempt < retries:
            delay = min(
                30.0,
                retry_delay * (2 ** (attempt - 1)),
            )
            delay += (job.index % 7) * 0.15
            time.sleep(delay)

    return DownloadResult(
        job=job,
        success=False,
        skipped=False,
        bytes_written=0,
        elapsed=time.monotonic() - started,
        attempts=retries,
        error=last_error,
    )


def print_result(
    result: DownloadResult,
    output_root: Path,
) -> None:
    """Print one compact, aligned download result."""
    status = "✓" if result.success else "✗"

    rate = (
        result.bytes_written / result.elapsed
        if result.elapsed > 0
        else 0
    )

    remaining = free_space(output_root)

    label = (
        f"{result.job.track.artist} - "
        f"{result.job.track.title}"
    )

    total = (
        str(result.job.total)
        if result.job.total
        else "?"
    )

    print(
        f"  [{result.job.index:>4}/{total:<4}] "
        f"{status} "
        f"{compact(label, 55):<55} "
        f"{human_size(result.bytes_written):>10} "
        f"{human_rate(rate):>12} "
        f"{human_size(remaining):>10}"
    )

    if result.skipped:
        print("       ↳ already downloaded")

    if not result.success:
        print(
            f"       ! {compact(' '.join(result.error.split()), 140)}"
        )


def create_jobs(
    tracks: list[Track],
    playlist_name: str,
    output_root: Path,
    directory_limit: int,
    supported_formats: set[str],
    conversion: Conversion,
) -> list[DownloadJob]:
    """Create deterministic download jobs."""
    jobs = []

    for index, track in enumerate(tracks, 1):
        if direct_copy_allowed(
            track,
            supported_formats,
        ):
            extension = source_format(track)
        else:
            extension = conversion.format

        destination = build_output_path(
            root=output_root,
            playlist_name=playlist_name,
            position=index,
            track=track,
            directory_limit=directory_limit,
            extension=extension,
        )

        jobs.append(
            DownloadJob(
                index=index,
                total=len(tracks),
                track=track,
                destination=destination,
                output_format=extension,
            )
        )

    return jobs


def download_playlist(
    tracks: list[Track],
    playlist_name: str,
    output_root: Path,
    directory_limit: int,
    server: PlexServer,
    supported_formats: set[str],
    conversion: Conversion,
    retries: int,
    retry_delay: float,
    timeout: int,
) -> tuple[int, int, int]:
    """Download all tracks from one playlist in parallel."""
    if not tracks:
        return 0, 0, 0

    jobs = create_jobs(
        tracks,
        playlist_name,
        output_root,
        directory_limit,
        supported_formats,
        conversion,
    )

    workers = determine_workers()

    print()
    print(f"  {playlist_name}")
    print(f"  {'─' * min(72, len(playlist_name) + 2)}")
    print(f"  Tracks:     {len(jobs):,}")
    print(f"  Parallel:   {workers}")
    print(
        f"  Conversion: {conversion.format}:{conversion.quality}"
    )
    print()

    downloaded = 0
    failed = 0
    written = 0

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="plex",
    ) as executor:
        futures = [
            executor.submit(
                download_track,
                job,
                server,
                supported_formats,
                conversion,
                retries,
                retry_delay,
                timeout,
            )
            for job in jobs
        ]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()

            if result.success:
                downloaded += 1
                written += result.bytes_written
            else:
                failed += 1

            print_result(
                result,
                output_root,
            )

    return downloaded, failed, written


def download_random_fill(
    tracks: list[Track],
    output_root: Path,
    directory_limit: int,
    server: PlexServer,
    supported_formats: set[str],
    conversion: Conversion,
    retries: int,
    retry_delay: float,
    timeout: int,
) -> tuple[int, int, int]:
    """Fill available USB capacity with random new music."""
    candidates = select_random_tracks(
        tracks,
        output_root,
    )

    if not candidates:
        print()
        print("  Random: no new tracks remain.")
        return 0, 0, 0

    workers = determine_workers()

    print()
    print("  Random")
    print("  ──────")
    print(f"  Library tracks:   {len(tracks):,}")
    print(f"  New candidates:   {len(candidates):,}")
    print(f"  Parallel:         {workers}")
    print(
        f"  Conversion:       {conversion.format}:{conversion.quality}"
    )
    print("  Fill mode:        until drive capacity")
    print(
        f"  Safety reserve:   {human_size(RANDOM_FILL_RESERVE)}"
    )
    print()

    downloaded = 0
    failed = 0
    written = 0

    random_root = output_root / playlist_directory("Random")

    existing_count = (
        sum(
            1
            for path in random_root.rglob("*")
            if path.is_file()
        )
        if random_root.exists()
        else 0
    )

    position = existing_count + 1
    cursor = 0

    while (
        cursor < len(candidates)
        and random_fill_space_available(output_root)
    ):
        batch = candidates[cursor : cursor + workers]
        cursor += len(batch)

        jobs = []

        for track in batch:
            extension = (
                source_format(track)
                if direct_copy_allowed(
                    track,
                    supported_formats,
                )
                else conversion.format
            )

            destination = build_output_path(
                root=output_root,
                playlist_name="Random",
                position=position,
                track=track,
                directory_limit=directory_limit,
                extension=extension,
            )

            jobs.append(
                DownloadJob(
                    index=position,
                    total=0,
                    track=track,
                    destination=destination,
                    output_format=extension,
                )
            )

            position += 1

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="random",
        ) as executor:
            futures = [
                executor.submit(
                    download_track,
                    job,
                    server,
                    supported_formats,
                    conversion,
                    retries,
                    retry_delay,
                    timeout,
                )
                for job in jobs
            ]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                print_result(
                    result,
                    output_root,
                )

                if result.success:
                    downloaded += 1
                    written += result.bytes_written
                else:
                    failed += 1

        if not random_fill_space_available(output_root):
            break

    remaining = free_space(output_root)

    print()
    print(
        f"  Random fill stopped with "
        f"{human_size(remaining)} remaining."
    )

    if cursor >= len(candidates):
        print(
            "  Random fill exhausted all "
            "available new library tracks."
        )

    return downloaded, failed, written


def main() -> int:
    """Run the interactive Plex music download workflow."""
    print()
    print("=" * 78)
    print("                              plexamp-usb")
    print("=" * 78)
    print()
    print("  Export Plex music to a car-friendly USB filesystem.")
    print()

    try:
        config = load_config()
        conversion = validate_audio_config(config)

        supported_formats = set(
            config["audio"]["supported_formats"]
        )

        server = prompt_server(config)

        timeout = int(
            config["plex"].get("timeout", 30)
        )

        library_key, section_name = select_music_library(
            server,
            timeout,
        )

        print()
        print(f"Music library: {section_name}")

        playlists = get_playlists(
            server,
            timeout,
        )

        selected = choose_playlists(playlists)

        directory_limit = choose_directory_limit(config)

        configured_output = (
            str(
                config["output"].get(
                    "directory",
                    DOWNLOAD_DIR,
                )
            ).strip()
            or DOWNLOAD_DIR
        )

        output_root = (
            Path(__file__).resolve().parent
            / configured_output
        )

        retries = max(
            1,
            int(
                config["download"].get(
                    "retries",
                    5,
                )
            ),
        )

        retry_delay = max(
            0.1,
            float(
                config["download"].get(
                    "retry_delay",
                    2.0,
                )
            ),
        )

        config["download"]["directory_limit"] = directory_limit
        save_config(config)

        has_random = any(
            title.casefold().strip() == "random"
            for _, title in selected
        )

        print()
        print("Download configuration")
        print()
        print(f"  Server:       {server.name}")
        print(f"  Address:      {server.base_url}")
        print(
            f"  Supported:    {', '.join(sorted(supported_formats))}"
        )
        print(
            f"  Conversion:   {conversion.format}:{conversion.quality}"
        )
        print(f"  Parallel:     {determine_workers()} workers")
        print(f"  Retries:      {retries}")
        print(
            "  Directory:    "
            + (
                "unlimited"
                if directory_limit == -1
                else str(directory_limit)
            )
        )
        print(f"  Destination:  {output_root}")

        if has_random:
            print("  Random fill:  enabled")

        print()

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        selected_tracks = []
        random_selected = False

        print("Collecting music...")
        print()

        for playlist_id, playlist_name in selected:
            if playlist_name.casefold().strip() == "random":
                random_selected = True
                continue

            print(
                f"  {compact(playlist_name, 60):<60}",
                end=" ",
                flush=True,
            )

            tracks = unique_tracks(
                fetch_playlist_tracks(
                    server,
                    playlist_id,
                    timeout,
                )
            )

            selected_tracks.append(
                (playlist_name, tracks)
            )

            print(f"{len(tracks):>6,} tracks")

        random_tracks = []

        if random_selected:
            print(
                "  Random"
                f"{'':<54}",
                end=" ",
                flush=True,
            )

            random_tracks = unique_tracks(
                fetch_library_tracks(
                    server,
                    library_key,
                    timeout,
                )
            )

            print(
                f"{len(random_tracks):>6,} tracks"
            )

        # FFmpeg is only required when at least one selected source needs
        # conversion. Supported source formats are copied directly.
        all_tracks = [
            track
            for _, tracks in selected_tracks
            for track in tracks
        ] + random_tracks

        needs_conversion = any(
            not direct_copy_allowed(
                track,
                supported_formats,
            )
            for track in all_tracks
        )

        if needs_conversion:
            check_ffmpeg()

        print()

        grand_downloaded = 0
        grand_failed = 0
        grand_written = 0

        for playlist_name, tracks in selected_tracks:
            if not tracks:
                print(
                    f"  Skipping empty download: {playlist_name}"
                )
                continue

            downloaded, failed, written = download_playlist(
                tracks=tracks,
                playlist_name=playlist_name,
                output_root=output_root,
                directory_limit=directory_limit,
                server=server,
                supported_formats=supported_formats,
                conversion=conversion,
                retries=retries,
                retry_delay=retry_delay,
                timeout=timeout,
            )

            grand_downloaded += downloaded
            grand_failed += failed
            grand_written += written

        # Random is deliberately last so it consumes only capacity left after
        # explicitly requested playlists have been exported.
        if random_selected:
            downloaded, failed, written = download_random_fill(
                tracks=random_tracks,
                output_root=output_root,
                directory_limit=directory_limit,
                server=server,
                supported_formats=supported_formats,
                conversion=conversion,
                retries=retries,
                retry_delay=retry_delay,
                timeout=timeout,
            )

            grand_downloaded += downloaded
            grand_failed += failed
            grand_written += written

        remaining = free_space(output_root)

        print()
        print("─" * 78)
        print("  Music download complete")
        print("─" * 78)
        print(
            f"  Current                 "
            f"{grand_downloaded + grand_failed:,}"
        )
        print(
            f"  Downloaded              "
            f"{grand_downloaded:,}"
        )
        print(
            f"  Failed                  "
            f"{grand_failed:,}"
        )
        print(
            f"  Written                 "
            f"{human_size(grand_written):>10}"
        )
        print(
            f"  Space remaining         "
            f"{human_size(remaining):>10}"
        )
        print(
            f"  Location                "
            f"{output_root}"
        )
        print()

        if grand_failed:
            print("  ! Some downloads failed.")
            print("    Run again to retry them.")
            print()
            return 2

        print("  ✓ All downloads completed successfully.")
        print()

        return 0

    except KeyboardInterrupt:
        # Keep the abort notice on the current status line before presenting
        # the final interrupted state.
        print("\r  Aborting...".ljust(78), end="", flush=True)
        print("\r  Interrupted.".ljust(78))
        return 130

    except Exception as exc:
        print()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
