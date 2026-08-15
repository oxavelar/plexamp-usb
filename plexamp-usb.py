#!/usr/bin/env python3
"""Export Plex music to a car-friendly USB filesystem.

The program discovers a reachable Plex Media Server, authenticates only when
local unauthenticated access is unavailable, lets the user select Plex music
playlists, and exports tracks to a configurable audio format.

Downloads are deterministic and resumable. Completed files are retained only
when they pass media validation, including duration validation against the
Plex track metadata. Incomplete or invalid files are replaced automatically.

A playlist named "Random" is treated as a special fill mode. When selected,
tracks are sampled from the complete Plex music library and downloaded until
the destination filesystem reaches its safety threshold.

Configuration is stored in settings.json beside this script. The generated
settings become the defaults for subsequent runs.

Requirements:
    - Python 3.10+
    - ffmpeg
    - ffprobe is preferred for fast duration validation; ffmpeg is used as
      the fallback when ffprobe is unavailable.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import unicodedata


APP_NAME = "plexamp-usb"
CONFIG_JSON = "settings.json"
DOWNLOAD_DIR = "Downloads"

# Keep enough space for filesystem metadata and the final in-flight writes.
RANDOM_FILL_RESERVE = 128 * 1024 * 1024

# A small tolerance avoids rejecting otherwise correct files because of
# encoder/container rounding or minor differences in reported timestamps.
DURATION_TOLERANCE_SECONDS = 2.0

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
        "conversion_formats": ["mp3:V0"],
        "conversion_threads": "auto",
    },
    "download": {
        "retries": 10,
        "retry_delay": 2.0,
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
    """Fit text into a fixed console width."""
    text = str(text)

    if len(text) <= width:
        return text

    if width <= 1:
        return text[:width]

    return text[: width - 1] + "…"


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
    """Format milliseconds for console display."""
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


def sanitize_filename(
    value: str,
    fallback: str = "Unknown",
    max_bytes: int = 255,
) -> str:
    """Return a deterministic filesystem-safe filename component."""
    value = str(value or "")
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\x00", "")

    value = re.sub(
        r'[<>:"/\\|?*\x00-\x1f\x7f]',
        "_",
        value,
    )

    value = re.sub(r"\s+", " ", value).strip()
    value = value.rstrip(". ")

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


def stable_hash(value: str, length: int = 8) -> str:
    """Return a short deterministic hash."""
    return hashlib.sha1(
        value.encode("utf-8")
    ).hexdigest()[:length]


def track_filename(track: Track, number: int) -> str:
    """Build a stable car-friendly filename."""
    artist = sanitize_filename(
        track.artist,
        "Unknown Artist",
    )
    album = sanitize_filename(
        track.album,
        "Unknown Album",
    )
    title = sanitize_filename(
        track.title,
        "Unknown Track",
    )

    try:
        track_number = int(track.index)
    except (TypeError, ValueError):
        track_number = 0

    prefix = (
        f"{track_number:02d}"
        if track_number
        else f"{number:02d}"
    )

    extension = ".mp3"
    prefix_part = (
        f"{prefix} - {artist} - {album} - "
    )

    available = (
        255
        - len(
            (prefix_part + extension).encode("utf-8")
        )
    )

    if available <= 0:
        album = sanitize_filename(
            album,
            "Unknown Album",
            max_bytes=80,
        )
        prefix_part = (
            f"{prefix} - {artist} - {album} - "
        )
        available = (
            255
            - len(
                (prefix_part + extension).encode("utf-8")
            )
        )

    title = sanitize_filename(
        title,
        "Unknown Track",
        max_bytes=max(1, available),
    )

    return f"{prefix_part}{title}{extension}"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge user configuration over defaults."""
    result = dict(base)

    for key, value in override.items():
        if (
            isinstance(result.get(key), dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(
                result[key],
                value,
            )
        else:
            result[key] = value

    return result


def save_default_config(path: Path) -> None:
    """Write the initial configuration."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            DEFAULT_CONFIG,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")


def load_config() -> dict:
    """Load settings.json, creating it when absent."""
    path = (
        Path(__file__).resolve().parent
        / CONFIG_JSON
    )

    if not path.exists():
        save_default_config(path)
        print(f"  Created {path}")
        return deep_merge({}, DEFAULT_CONFIG)

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            loaded = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON in {path}: {exc}"
        ) from exc

    if not isinstance(loaded, dict):
        raise RuntimeError(
            f"Invalid configuration: {path}"
        )

    return deep_merge(
        DEFAULT_CONFIG,
        loaded,
    )


def validate_config(config: dict) -> None:
    """Validate the small set of configuration values used at runtime."""
    audio = config.get("audio", {})

    formats = audio.get(
        "conversion_formats",
        [],
    )

    if not isinstance(formats, list) or not formats:
        raise RuntimeError(
            "audio.conversion_formats must contain at least one format."
        )

    for value in formats:
        if (
            not isinstance(value, str)
            or ":" not in value
        ):
            raise RuntimeError(
                "audio.conversion_formats must contain "
                'values such as "mp3:V0".'
            )

    threads = audio.get(
        "conversion_threads",
        "auto",
    )

    if threads != "auto":
        try:
            if int(threads) < 1:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "audio.conversion_threads must be "
                '"auto" or a positive integer.'
            ) from exc

    download = config.get(
        "download",
        {},
    )

    try:
        if int(download.get("retries", 10)) < 1:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "download.retries must be a positive integer."
        ) from exc


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
        accept=(
            "application/xml,"
            "application/json,"
            "text/plain,*/*"
        ),
    )

    with urllib.request.urlopen(
        request,
        timeout=timeout,
    ) as response:
        return response.read()


def plex_xml(
    server: PlexServer,
    path: str,
    params: dict | None = None,
    timeout: int = 30,
) -> ET.Element:
    """Request and parse one Plex XML endpoint."""
    query = urllib.parse.urlencode(
        params or {}
    )

    url = f"{server.base_url}{path}"

    if query:
        url = f"{url}?{query}"

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
    discovered: dict[
        tuple[str, int],
        PlexServer,
    ] = {}

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

            if not data:
                continue

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

            if (
                "plex/media-server"
                not in headers.get(
                    "content-type",
                    "",
                ).lower()
            ):
                continue

            host = (
                headers.get("host", "").strip()
                or address[0]
            )

            if host.endswith(".plex.direct"):
                host = address[0]

            try:
                port = int(
                    headers.get(
                        "port",
                        "32400",
                    )
                )
            except ValueError:
                port = 32400

            name = headers.get(
                "name",
                host,
            )

            discovered[(host, port)] = PlexServer(
                name=name,
                host=host,
                port=port,
            )

    finally:
        sock.close()

    return list(discovered.values())


def prompt_server(config: dict) -> PlexServer:
    """Select a Plex server, preferring local access."""
    plex = config["plex"]

    timeout = int(
        plex.get("timeout", 30)
    )
    configured_host = str(
        plex.get("host") or ""
    ).strip()
    configured_port = int(
        plex.get("port") or 32400
    )
    configured_token = str(
        plex.get("token") or ""
    ).strip()

    if configured_host:
        configured = PlexServer(
            name=configured_host,
            host=configured_host,
            port=configured_port,
            token=configured_token,
        )

        identity = test_server(
            configured,
            timeout,
        )

        if identity is not None:
            name = identity.attrib.get(
                "friendlyName",
                configured_host,
            )

            print(f"  Server: {name}")
            print(f"  Address: {configured.base_url}")
            print("  Access:  OK")

            return PlexServer(
                name=name,
                host=configured_host,
                port=configured_port,
                token=configured_token,
            )

        print(
            f"  Configured server unavailable: "
            f"{configured.base_url}"
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

        host = input(
            "  Plex server address: "
        ).strip()

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
                possible_host, possible_port = host.rsplit(
                    ":",
                    1,
                )

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

        for index, candidate in enumerate(
            servers,
            1,
        ):
            print(
                f"    {index}. "
                f"{candidate.name} "
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
        f"  Found: {server.name} "
        f"({server.base_url})"
    )

    identity = test_server(
        server,
        timeout,
    )

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
        token = input(
            "  Plex token: "
        ).strip()

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

    identity = test_server(
        authenticated,
        timeout,
    )

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

    libraries: list[tuple[str, str]] = []

    for directory in root.findall("Directory"):
        if directory.attrib.get("type") != "artist":
            continue

        key = directory.attrib.get("key", "")
        title = directory.attrib.get(
            "title",
            "Music",
        )

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

    for index, (_, title) in enumerate(
        libraries,
        1,
    ):
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
    """Return Plex audio playlists."""
    root = plex_xml(
        server,
        "/playlists",
        params={"playlistType": "audio"},
        timeout=timeout,
    )

    playlists = []

    for playlist in root.findall("Playlist"):
        rating_key = playlist.attrib.get(
            "ratingKey",
            "",
        )

        title = playlist.attrib.get(
            "title",
            "Unnamed",
        )

        if not rating_key:
            continue

        try:
            count = int(
                playlist.attrib.get(
                    "leafCount",
                    0,
                )
            )
        except ValueError:
            count = 0

        duration = human_duration(
            playlist.attrib.get("duration")
        )

        playlists.append(
            (
                rating_key,
                title,
                count,
                duration,
            )
        )

    return playlists


def choose_playlists(
    playlists: list[tuple[str, str, int, str]],
) -> list[tuple[str, str]]:
    """Prompt for one or more playlist downloads."""
    if not playlists:
        raise RuntimeError(
            "No Plex music playlists were found."
        )

    print()
    print("Music playlists:")
    print()

    for index, (
        _,
        title,
        count,
        duration,
    ) in enumerate(
        playlists,
        1,
    ):
        print(
            f"  {index:>2}. "
            f"{title} "
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
                for rating_key, title, _, _
                in playlists
            ]

        try:
            indices: list[int] = []
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
                selected.append(
                    ("Random", "Random")
                )

            if not selected:
                raise ValueError

            return selected

        except ValueError:
            print("Invalid selection.")


def parse_track(item: ET.Element, server: PlexServer, playlist_id: str) -> Track | None:
    """Build a Track from one Plex track element."""
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
        source_size = int(
            part.attrib.get("size", 0)
        )
    except ValueError:
        source_size = 0

    try:
        duration_ms = int(
            item.attrib.get("duration", 0)
        )
    except ValueError:
        duration_ms = 0

    return Track(
        rating_key=item.attrib.get(
            "ratingKey",
            "",
        ),
        title=item.attrib.get(
            "title",
            "Unknown Track",
        ),
        artist=item.attrib.get(
            "grandparentTitle",
            item.attrib.get(
                "originalTitle",
                "Unknown Artist",
            ),
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
        index=item.attrib.get(
            "index",
            "0",
        ),
        duration_ms=duration_ms,
        media_url=urllib.parse.urljoin(
            server.base_url,
            key,
        ),
        source_size=source_size,
        playlist_id=playlist_id,
        container=part.attrib.get(
            "container",
            "",
        ),
        audio_codec=part.attrib.get(
            "audioCodec",
            "",
        ),
    )


def fetch_playlist_tracks(
    server: PlexServer,
    playlist_id: str,
    timeout: int,
) -> list[Track]:
    """Fetch playable tracks from a Plex playlist."""
    root = plex_xml(
        server,
        f"/playlists/{playlist_id}/items",
        timeout=timeout,
    )

    tracks = []

    for item in root:
        track = parse_track(
            item,
            server,
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
    """Fetch all playable tracks from a Plex music library."""
    root = plex_xml(
        server,
        f"/library/sections/{library_key}/all",
        params={"type": 10},
        timeout=timeout,
    )

    tracks = []

    for item in root.findall("Track"):
        track = parse_track(
            item,
            server,
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
    seen: set[str] = set()

    for track in tracks:
        identity = (
            track.rating_key
            or (
                f"{track.media_url}|"
                f"{track.title}|"
                f"{track.artist}|"
                f"{track.album}"
            )
        )

        if identity in seen:
            continue

        seen.add(identity)
        result.append(track)

    return result


def choose_directory_limit() -> int:
    """Prompt for the maximum number of files per directory."""
    print()
    print("Directory limit")
    print()
    print("  - Maximum 255 audio files per directory.")
    print("  - Enter -1 for unlimited.")
    print()

    while True:
        answer = input(
            "  Maximum files per directory [255]: "
        ).strip()

        if not answer:
            return 255

        try:
            value = int(answer)

            if value == -1 or value > 0:
                return value
        except ValueError:
            pass

        print("  Enter a positive number or -1.")


def playlist_directory(playlist_name: str) -> str:
    """Return a safe playlist directory name."""
    return sanitize_filename(
        playlist_name,
        "Music",
    )


def build_output_path(
    root: Path,
    playlist_name: str,
    position: int,
    track: Track,
    directory_limit: int,
) -> Path:
    """Return the deterministic destination for one track."""
    playlist_root = (
        root
        / playlist_directory(playlist_name)
    )

    if directory_limit == -1:
        directory = playlist_root
        directory_position = position
    else:
        directory_number = (
            (position - 1) // directory_limit
        ) + 1

        directory_position = (
            (position - 1) % directory_limit
        ) + 1

        directory = (
            playlist_root
            / f"{directory_number:03d}"
        )

    return directory / track_filename(
        track,
        directory_position,
    )


def check_ffmpeg() -> None:
    """Verify that FFmpeg is available."""
    try:
        completed = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(
            "ffmpeg is required but was not found in PATH."
        ) from exc

    if completed.returncode != 0:
        raise RuntimeError(
            "ffmpeg could not be executed."
        )


def ffprobe_duration(path: Path) -> float | None:
    """Return the decoded media duration when ffprobe is available."""
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
            check=False,
        )
    except (
        OSError,
        subprocess.TimeoutExpired,
    ):
        return None

    if completed.returncode != 0:
        return None

    try:
        duration = float(
            completed.stdout.strip()
        )
    except ValueError:
        return None

    return duration if duration > 0 else None


def ffmpeg_duration(path: Path) -> float | None:
    """Decode the complete file and return its actual audio duration."""
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-f",
        "null",
        "-",
    ]

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
            check=False,
        )
    except (
        OSError,
        subprocess.TimeoutExpired,
    ):
        return None

    # FFmpeg's null decode is deliberately treated as the authoritative
    # fallback because it reads through the entire audio stream.
    if completed.returncode != 0:
        return None

    # A successful decode does not directly expose the final timestamp with
    # error-level logging, so use ffprobe if possible and otherwise report
    # validation failure rather than accepting an uncertain file.
    return ffprobe_duration(path)


def actual_duration(path: Path) -> float | None:
    """Return a reliable duration or None when validation cannot confirm it."""
    duration = ffprobe_duration(path)

    if duration is not None:
        return duration

    return ffmpeg_duration(path)


def duration_matches(
    path: Path,
    expected_ms: int,
) -> tuple[bool, str]:
    """Check the completed file against Plex's expected duration."""
    if expected_ms <= 0:
        return True, ""

    actual = actual_duration(path)

    if actual is None:
        return False, "unable to determine output duration"

    expected = expected_ms / 1000.0
    difference = abs(actual - expected)

    if difference <= DURATION_TOLERANCE_SECONDS:
        return True, ""

    return (
        False,
        "duration mismatch: "
        f"got {human_duration(actual * 1000)}, "
        f"expected {human_duration(expected_ms)}",
    )


def file_is_valid(
    path: Path,
    expected_duration_ms: int = 0,
) -> tuple[bool, str]:
    """Validate size, decodability, and expected duration."""
    try:
        if not path.is_file():
            return False, "file does not exist"

        if path.stat().st_size < 1024:
            return False, "file is too small"
    except OSError:
        return False, "unable to inspect file"

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-f",
        "null",
        "-",
    ]

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
            check=False,
        )
    except (
        OSError,
        subprocess.TimeoutExpired,
    ):
        return False, "audio validation failed"

    if completed.returncode != 0:
        return False, "audio decode failed"

    return duration_matches(
        path,
        expected_duration_ms,
    )


def determine_conversion_threads(config: dict) -> int:
    """Return the configured maximum number of FFmpeg conversions."""
    value = config["audio"].get(
        "conversion_threads",
        "auto",
    )

    if value == "auto":
        return os.cpu_count() or 1

    return max(1, int(value))


def determine_queue_size(
    conversion_threads: int,
) -> int:
    """Keep USB/network work queued without multiplying FFmpeg processes."""
    return max(
        conversion_threads * 4,
        16,
    )


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
    return (
        free_space(output_root)
        > RANDOM_FILL_RESERVE
    )


def exported_track_keys(
    output_root: Path,
) -> set[str]:
    """Build normalized identities from exported filenames."""
    keys: set[str] = set()

    if not output_root.exists():
        return keys

    for path in output_root.rglob("*.mp3"):
        parts = path.stem.split(" - ", 3)

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
    """Return a normalized track identity."""
    return "|".join(
        (
            sanitize_filename(
                track.artist,
                "",
            ).casefold(),
            sanitize_filename(
                track.album,
                "",
            ).casefold(),
            sanitize_filename(
                track.title,
                "",
            ).casefold(),
        )
    )


def select_random_tracks(
    tracks: list[Track],
    output_root: Path,
) -> list[Track]:
    """Shuffle library tracks and remove obvious exported duplicates."""
    existing = exported_track_keys(output_root)

    candidates = [
        track
        for track in tracks
        if track_identity(track) not in existing
    ]

    random.SystemRandom().shuffle(candidates)

    return candidates


def parse_conversion(value: str) -> tuple[str, str]:
    """Parse a conversion specification such as mp3:V0."""
    try:
        container, quality = value.split(":", 1)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid conversion format: {value!r}"
        ) from exc

    container = container.strip().lower()
    quality = quality.strip()

    if not container or not quality:
        raise RuntimeError(
            f"Invalid conversion format: {value!r}"
        )

    return container, quality


def choose_conversion(
    config: dict,
) -> tuple[str, str]:
    """Return the first configured conversion target."""
    formats = config["audio"]["conversion_formats"]

    # The format list is intentionally simple for now. Keeping the parser
    # separate allows additional format specifications without changing the
    # download pipeline.
    return parse_conversion(formats[0])


def source_matches_format(
    track: Track,
    output_format: str,
) -> bool:
    """Return whether the source can be copied without conversion."""
    return (
        track.container.casefold() == output_format
        and track.audio_codec.casefold() == output_format
    ) or (
        output_format == "mp3"
        and (
            track.container.casefold() == "mp3"
            or track.audio_codec.casefold() == "mp3"
            or track.media_url.casefold().split("?", 1)[0].endswith(
                ".mp3"
            )
        )
    )


def ffmpeg_command(
    url: str,
    output: Path,
    token: str,
    output_format: str,
    quality: str,
    copy: bool,
) -> list[str]:
    """Build the FFmpeg command for one transfer."""
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",

        # Retry transient HTTP failures while FFmpeg is reading from Plex.
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

    if copy:
        command.extend(
            [
                "-c:a",
                "copy",
                "-f",
                output_format,
                str(output),
            ]
        )
        return command

    if output_format == "mp3":
        command.extend(
            [
                "-c:a",
                "libmp3lame",
                "-q:a",
                quality.lower().removeprefix("v"),
                "-f",
                "mp3",
                str(output),
            ]
        )
        return command

    raise RuntimeError(
        f"Unsupported conversion format: "
        f"{output_format}:{quality}"
    )


def run_ffmpeg(
    track: Track,
    destination: Path,
    token: str,
    timeout: int,
    output_format: str,
    quality: str,
    copy: bool,
) -> tuple[bool, int, str]:
    """Transfer one track into an atomic temporary file."""
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
        url=track.media_url,
        output=part_path,
        token=token,
        output_format=output_format,
        quality=quality,
        copy=copy,
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
            or (
                "ffmpeg exited with code "
                f"{completed.returncode}"
            )
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

        return False, 0, "ffmpeg produced an empty file"

    valid, error = file_is_valid(
        part_path,
        track.duration_ms,
    )

    if not valid:
        try:
            part_path.unlink(missing_ok=True)
        except OSError:
            pass

        return False, 0, error

    try:
        os.replace(
            part_path,
            destination,
        )
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
    retries: int,
    retry_delay: float,
    timeout: int,
    output_format: str,
    quality: str,
) -> DownloadResult:
    """Download one track, retrying incomplete streams automatically."""
    destination = job.destination

    valid, _ = file_is_valid(
        destination,
        job.track.duration_ms,
    )

    if valid:
        try:
            size = destination.stat().st_size
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

    copy = source_matches_format(
        job.track,
        output_format,
    )

    for attempt in range(1, retries + 1):
        success, size, error = run_ffmpeg(
            track=job.track,
            destination=destination,
            token=server.token,
            timeout=timeout,
            output_format=output_format,
            quality=quality,
            copy=copy,
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

        if attempt >= retries:
            break

        delay = min(
            30.0,
            retry_delay * (2 ** (attempt - 1)),
        )

        delay += (
            job.index % 7
        ) * 0.15

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
    """Print one compact download result."""
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
            "       ! "
            + compact(
                " ".join(result.error.split()),
                140,
            )
        )


def create_jobs(
    tracks: list[Track],
    playlist_name: str,
    output_root: Path,
    directory_limit: int,
) -> list[DownloadJob]:
    """Create deterministic download jobs."""
    total = len(tracks)

    return [
        DownloadJob(
            index=index,
            total=total,
            track=track,
            destination=build_output_path(
                root=output_root,
                playlist_name=playlist_name,
                position=index,
                track=track,
                directory_limit=directory_limit,
            ),
        )
        for index, track in enumerate(
            tracks,
            1,
        )
    ]


def download_playlist(
    tracks: list[Track],
    playlist_name: str,
    output_root: Path,
    directory_limit: int,
    server: PlexServer,
    retries: int,
    retry_delay: float,
    timeout: int,
    conversion_threads: int,
    queue_size: int,
    output_format: str,
    quality: str,
) -> tuple[int, int, int]:
    """Download a playlist with bounded conversion concurrency."""
    if not tracks:
        return 0, 0, 0

    jobs = create_jobs(
        tracks,
        playlist_name,
        output_root,
        directory_limit,
    )

    print()
    print(f"  {playlist_name}")
    print(
        f"  {'─' * min(72, len(playlist_name) + 2)}"
    )
    print(f"  Tracks:          {len(jobs):,}")
    print(f"  Conversion:      {output_format}:{quality}")
    print(f"  Conversion jobs: {conversion_threads}")
    print(f"  Queue:           {queue_size}")
    print()

    downloaded = 0
    failed = 0
    written = 0

    # The queue is intentionally deeper than the conversion pool. This lets
    # USB/network latency stay hidden without spawning excessive FFmpeg jobs.
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=conversion_threads,
        thread_name_prefix="plex",
    ) as executor:
        pending: set[
            concurrent.futures.Future[DownloadResult]
        ] = set()

        iterator = iter(jobs)

        while pending or jobs:
            while len(pending) < queue_size:
                try:
                    job = next(iterator)
                except StopIteration:
                    break

                pending.add(
                    executor.submit(
                        download_track,
                        job,
                        server,
                        retries,
                        retry_delay,
                        timeout,
                        output_format,
                        quality,
                    )
                )

            if not pending:
                break

            done, pending = (
                concurrent.futures.wait(
                    pending,
                    return_when=(
                        concurrent.futures.FIRST_COMPLETED
                    ),
                )
            )

            for future in done:
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
    retries: int,
    retry_delay: float,
    timeout: int,
    conversion_threads: int,
    queue_size: int,
    output_format: str,
    quality: str,
) -> tuple[int, int, int]:
    """Fill remaining USB capacity with random new music."""
    candidates = select_random_tracks(
        tracks,
        output_root,
    )

    if not candidates:
        print()
        print("  Random: no new tracks remain.")
        return 0, 0, 0

    random_root = (
        output_root
        / playlist_directory("Random")
    )

    existing_count = (
        sum(
            1
            for path in random_root.rglob("*.mp3")
        )
        if random_root.exists()
        else 0
    )

    print()
    print("  Random")
    print("  ──────")
    print(f"  Library tracks:   {len(tracks):,}")
    print(f"  New candidates:   {len(candidates):,}")
    print(f"  Conversion jobs:  {conversion_threads}")
    print(f"  Queue:            {queue_size}")
    print(
        f"  Conversion:       {output_format}:{quality}"
    )
    print(
        f"  Safety reserve:   "
        f"{human_size(RANDOM_FILL_RESERVE)}"
    )
    print()

    downloaded = 0
    failed = 0
    written = 0
    position = existing_count + 1
    cursor = 0

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=conversion_threads,
        thread_name_prefix="random",
    ) as executor:
        pending: set[
            concurrent.futures.Future[DownloadResult]
        ] = set()

        while (
            cursor < len(candidates)
            or pending
        ):
            while (
                cursor < len(candidates)
                and len(pending) < queue_size
                and random_fill_space_available(
                    output_root
                )
            ):
                track = candidates[cursor]
                cursor += 1

                destination = build_output_path(
                    root=output_root,
                    playlist_name="Random",
                    position=position,
                    track=track,
                    directory_limit=directory_limit,
                )

                job = DownloadJob(
                    index=position,
                    total=0,
                    track=track,
                    destination=destination,
                )

                pending.add(
                    executor.submit(
                        download_track,
                        job,
                        server,
                        retries,
                        retry_delay,
                        timeout,
                        output_format,
                        quality,
                    )
                )

                position += 1

            if not pending:
                break

            done, pending = (
                concurrent.futures.wait(
                    pending,
                    return_when=(
                        concurrent.futures.FIRST_COMPLETED
                    ),
                )
            )

            for future in done:
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

            if not random_fill_space_available(
                output_root
            ):
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
    """Run the interactive Plex music export workflow."""
    print()
    print("=" * 78)
    print(
        "                              plexamp-usb"
    )
    print("=" * 78)
    print()
    print(
        "  Export Plex music to a car-friendly USB filesystem."
    )
    print()

    try:
        check_ffmpeg()

        config = load_config()
        validate_config(config)

        server = prompt_server(config)

        timeout = int(
            config["plex"].get(
                "timeout",
                30,
            )
        )

        library_key, section_name = (
            select_music_library(
                server,
                timeout,
            )
        )

        print()
        print(f"Music library: {section_name}")

        playlists = get_playlists(
            server,
            timeout,
        )

        selected = choose_playlists(
            playlists
        )

        directory_limit = (
            choose_directory_limit()
        )

        configured_output = str(
            config["output"].get(
                "directory",
                DOWNLOAD_DIR,
            )
        ).strip() or DOWNLOAD_DIR

        output_root = (
            Path(__file__).resolve().parent
            / configured_output
        )

        retries = max(
            1,
            int(
                config["download"].get(
                    "retries",
                    10,
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

        conversion_threads = (
            determine_conversion_threads(
                config
            )
        )

        queue_size = determine_queue_size(
            conversion_threads
        )

        output_format, quality = choose_conversion(
            config
        )

        has_random = any(
            title.casefold().strip() == "random"
            for _, title in selected
        )

        print()
        print("Download configuration")
        print()
        print(f"  Server:           {server.name}")
        print(f"  Address:          {server.base_url}")
        print(
            f"  Conversion:       {output_format}:{quality}"
        )
        print(
            f"  Conversion jobs:  {conversion_threads}"
        )
        print(
            f"  Queue:            {queue_size}"
        )
        print(
            f"  Retries:           {retries}"
        )
        print(
            "  Directory:        "
            + (
                "unlimited"
                if directory_limit == -1
                else str(directory_limit)
            )
        )
        print(f"  Destination:      {output_root}")

        if has_random:
            print("  Random fill:      enabled")

        print()

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        selected_tracks: list[
            tuple[str, list[Track]]
        ] = []

        random_selected = False

        print("Collecting music...")
        print()

        for playlist_id, playlist_name in selected:
            if (
                playlist_name.casefold().strip()
                == "random"
            ):
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
                (
                    playlist_name,
                    tracks,
                )
            )

            print(
                f"{len(tracks):>6,} tracks"
            )

        random_tracks: list[Track] = []

        if random_selected:
            print(
                f"  {'Random':<60}",
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

        print()

        grand_downloaded = 0
        grand_failed = 0
        grand_written = 0

        for playlist_name, tracks in selected_tracks:
            if not tracks:
                print(
                    f"  Skipping empty download: "
                    f"{playlist_name}"
                )
                continue

            downloaded, failed, written = (
                download_playlist(
                    tracks=tracks,
                    playlist_name=playlist_name,
                    output_root=output_root,
                    directory_limit=directory_limit,
                    server=server,
                    retries=retries,
                    retry_delay=retry_delay,
                    timeout=timeout,
                    conversion_threads=conversion_threads,
                    queue_size=queue_size,
                    output_format=output_format,
                    quality=quality,
                )
            )

            grand_downloaded += downloaded
            grand_failed += failed
            grand_written += written

        if random_selected:
            downloaded, failed, written = (
                download_random_fill(
                    tracks=random_tracks,
                    output_root=output_root,
                    directory_limit=directory_limit,
                    server=server,
                    retries=retries,
                    retry_delay=retry_delay,
                    timeout=timeout,
                    conversion_threads=conversion_threads,
                    queue_size=queue_size,
                    output_format=output_format,
                    quality=quality,
                )
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

        print(
            "  ✓ All downloads completed successfully."
        )
        print()

        return 0

    except KeyboardInterrupt:
        # Replace the transient abort message with the final state so an
        # interrupted run leaves a clean console and no misleading success.
        print(
            "\r  Aborting...          ",
            end="",
            flush=True,
        )
        time.sleep(0.15)
        print(
            "\r  Interrupted.         "
        )
        return 130

    except Exception as exc:
        print()
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
