#!/usr/bin/env python3
"""Export Plex music to a car-friendly USB filesystem.

The program discovers a reachable Plex Media Server, authenticates only when
local unauthenticated access is unavailable, lets the user select Plex music
playlists, converts unsupported audio to the configured target format, and
writes files below a Downloads directory beside this script.

Downloads are deterministic and resumable. Existing valid files are retained,
failed transfers are discarded, and later runs only process missing or invalid
files.

The conversion target is configured with entries such as ``mp3:V0``.
Already-supported source audio bypasses FFmpeg entirely.

Random mode samples new tracks from the complete Plex music library and fills
the destination until the configured filesystem safety reserve is reached.

Requirements:
    - Python 3.10+
    - ffmpeg
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

# Leave room for filesystem metadata and behavior while output sizes vary.
RANDOM_FILL_RESERVE = 128 * 1024 * 1024

# A deeper queue keeps Plex/network/USB activity moving while conversion
# concurrency remains bounded independently.
DEFAULT_QUEUE_SIZE = 64

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
        "conversion_formats": [
            "mp3:V0",
        ],
        "conversion_threads": "auto",
    },
    "download": {
        "retries": 10,
        "retry_delay": 2.0,
        "queue_size": DEFAULT_QUEUE_SIZE,
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
class ConversionFormat:
    """One configured output format."""

    container: str
    quality: str

    @property
    def spec(self) -> str:
        return f"{self.container}:{self.quality}"


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
    """Format bytes using binary units."""
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
    value = unicodedata.normalize("NFC", str(value or ""))
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


def track_filename(track: Track, number: int, extension: str) -> str:
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

    prefix_part = f"{prefix} - {artist} - {album} - "
    extension = extension if extension.startswith(".") else f".{extension}"

    available = 255 - len(
        (prefix_part + extension).encode("utf-8")
    )

    if available <= 0:
        album = sanitize_filename(
            album,
            "Unknown Album",
            max_bytes=80,
        )
        prefix_part = f"{prefix} - {artist} - {album} - "
        available = 255 - len(
            (prefix_part + extension).encode("utf-8")
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

        return deep_merge(
            {},
            DEFAULT_CONFIG,
        )

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


def parse_conversion_format(value: str) -> ConversionFormat:
    """Parse a format specification such as mp3:V0."""
    value = str(value).strip()

    if ":" not in value:
        raise RuntimeError(
            f"Invalid audio conversion format: {value!r}. "
            "Use format:quality, for example mp3:V0."
        )

    container, quality = value.split(
        ":",
        1,
    )

    container = container.strip().lower()
    quality = quality.strip()

    if not container or not quality:
        raise RuntimeError(
            f"Invalid audio conversion format: {value!r}."
        )

    return ConversionFormat(
        container=container,
        quality=quality,
    )


def get_conversion_formats(config: dict) -> list[ConversionFormat]:
    """Return configured conversion targets."""
    values = config["audio"].get(
        "conversion_formats",
        [],
    )

    if isinstance(values, str):
        values = [values]

    if not isinstance(values, list) or not values:
        raise RuntimeError(
            "audio.conversion_formats must contain at least one format."
        )

    formats = [
        parse_conversion_format(value)
        for value in values
    ]

    seen: set[str] = set()

    for conversion in formats:
        if conversion.spec in seen:
            continue

        seen.add(conversion.spec)

    return formats


def conversion_workers(value: str | int) -> int:
    """Return the configured FFmpeg concurrency."""
    if str(value).strip().lower() == "auto":
        return os.cpu_count() or 1

    try:
        workers = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "audio.conversion_threads must be "
            "'auto' or a positive integer."
        ) from exc

    if workers < 1:
        raise RuntimeError(
            "audio.conversion_threads must be "
            "'auto' or a positive integer."
        )

    return workers


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
    """Test Plex connectivity."""
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

                key, value = line.split(
                    ":",
                    1,
                )

                headers[key.strip().lower()] = value.strip()

            if (
                "plex/media-server"
                not in headers.get(
                    "content-type",
                    "",
                ).lower()
            ):
                continue

            host = headers.get(
                "host",
                "",
            ).strip() or address[0]

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

            discovered[
                (host, port)
            ] = PlexServer(
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

    timeout = int(
        plex.get(
            "timeout",
            30,
        )
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
            print(
                f"  Address: {configured.base_url}"
            )
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

        key = directory.attrib.get(
            "key",
            "",
        )
        title = directory.attrib.get(
            "title",
            "Music",
        )

        if key:
            libraries.append(
                (key, title)
            )

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
        print(
            f"  {index:>2}. {title}"
        )

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
        params={
            "playlistType": "audio",
        },
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
    """Prompt for one or more music playlists."""
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


def track_from_item(
    item: ET.Element,
    media: ET.Element,
    part: ET.Element,
    server: PlexServer,
    playlist_id: str,
) -> Track | None:
    """Build a Track from one Plex item."""
    key = part.attrib.get(
        "key",
        "",
    )

    if not key:
        return None

    try:
        source_size = int(
            part.attrib.get(
                "size",
                0,
            )
        )
    except ValueError:
        source_size = 0

    try:
        duration_ms = int(
            item.attrib.get(
                "duration",
                0,
            )
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
    """Fetch playable audio tracks from a playlist."""
    root = plex_xml(
        server,
        f"/playlists/{playlist_id}/items",
        timeout=timeout,
    )

    tracks = []

    for item in root:
        media = item.find("Media")

        if media is None:
            continue

        part = media.find("Part")

        if part is None:
            continue

        track = track_from_item(
            item,
            media,
            part,
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
    """Fetch all playable audio tracks from a music library."""
    root = plex_xml(
        server,
        f"/library/sections/{library_key}/all",
        params={
            "type": 10,
        },
        timeout=timeout,
    )

    tracks = []

    for item in root.findall("Track"):
        media = item.find("Media")

        if media is None:
            continue

        part = media.find("Part")

        if part is None:
            continue

        track = track_from_item(
            item,
            media,
            part,
            server,
            "Random",
        )

        if track is not None:
            tracks.append(track)

    return tracks


def unique_tracks(
    tracks: Iterable[Track],
) -> list[Track]:
    """Remove duplicate tracks while preserving order."""
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

            if value == -1:
                return -1

            if value > 0:
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
    extension: str,
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
            (position - 1)
            // directory_limit
        ) + 1

        directory_position = (
            (position - 1)
            % directory_limit
        ) + 1

        directory = (
            playlist_root
            / f"{directory_number:03d}"
        )

    return (
        directory
        / track_filename(
            track,
            directory_position,
            extension,
        )
    )


def file_is_valid(
    path: Path,
    expected_duration_ms: int = 0,
) -> tuple[bool, str]:
    """Validate an audio file and optionally verify its duration."""
    try:
        if not path.is_file():
            return False, "file does not exist"

        if path.stat().st_size < 1024:
            return False, "file is too small"
    except OSError:
        return False, "unable to inspect file"

    command = [
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
    ]

    try:
        completed = subprocess.run(
            command,
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
        return False, "audio validation failed"

    if completed.returncode != 0:
        return False, "audio validation failed"

    if expected_duration_ms <= 0:
        return True, ""

    try:
        actual_seconds = float(
            completed.stdout.strip()
        )
    except ValueError:
        return False, "audio duration unavailable"

    expected_seconds = expected_duration_ms / 1000

    # Container and encoder delay can move the reported duration slightly.
    if abs(actual_seconds - expected_seconds) > 2.0:
        return (
            False,
            "duration mismatch: "
            f"got {human_duration(actual_seconds * 1000)}, "
            f"expected {human_duration(expected_duration_ms)}",
        )

    return True, ""


def check_ffmpeg() -> None:
    """Verify FFmpeg is available."""
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-version",
            ],
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


def check_ffprobe() -> None:
    """Verify FFprobe is available for duration validation."""
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-version",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(
            "ffprobe is required but was not found in PATH."
        ) from exc

    if completed.returncode != 0:
        raise RuntimeError(
            "ffprobe could not be executed."
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
    """Return whether random fill can continue safely."""
    return (
        free_space(output_root)
        > RANDOM_FILL_RESERVE
    )


def exported_track_keys(
    output_root: Path,
) -> set[str]:
    """Build exported artist/album/title identities from filenames."""
    keys: set[str] = set()

    if not output_root.exists():
        return keys

    for path in output_root.rglob("*"):
        if path.suffix.lower() != ".mp3":
            continue

        parts = path.stem.split(
            " - ",
            3,
        )

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
    """Shuffle the library and remove already-exported tracks."""
    existing = exported_track_keys(
        output_root
    )

    candidates = [
        track
        for track in tracks
        if track_identity(track) not in existing
    ]

    random.SystemRandom().shuffle(candidates)

    return candidates


def source_matches_conversion(
    track: Track,
    conversion: ConversionFormat,
) -> bool:
    """Return whether the source can be copied without conversion."""
    container = (
        track.container or ""
    ).strip().lower()

    codec = (
        track.audio_codec or ""
    ).strip().lower()

    target = conversion.container.lower()

    # For the current target model, container/codec compatibility determines
    # whether FFmpeg is needed. Quality presets cannot be inferred reliably
    # from Plex metadata, so an already-compatible source is copied as-is.
    if target == "mp3":
        return (
            container == "mp3"
            or codec == "mp3"
            or track.media_url.lower().split("?")[0].endswith(
                ".mp3"
            )
        )

    return container == target or codec == target


def ffmpeg_quality_args(
    conversion: ConversionFormat,
) -> list[str]:
    """Translate the human quality value into FFmpeg arguments."""
    container = conversion.container.lower()
    quality = conversion.quality.strip()

    if container == "mp3":
        if quality.lower().startswith("v"):
            value = quality[1:]

            if not value.isdigit():
                raise RuntimeError(
                    f"Invalid MP3 VBR quality: {quality}"
                )

            return [
                "-c:a",
                "libmp3lame",
                "-q:a",
                value,
            ]

        if quality.lower().endswith("k"):
            bitrate = quality[:-1]

            if not bitrate.isdigit():
                raise RuntimeError(
                    f"Invalid MP3 bitrate: {quality}"
                )

            return [
                "-c:a",
                "libmp3lame",
                "-b:a",
                f"{bitrate}k",
            ]

        raise RuntimeError(
            f"Unsupported MP3 quality: {quality}"
        )

    if container in {"aac", "m4a"}:
        return [
            "-c:a",
            "aac",
            "-b:a",
            quality,
        ]

    if container == "opus":
        return [
            "-c:a",
            "libopus",
            "-b:a",
            quality,
        ]

    if container == "flac":
        return [
            "-c:a",
            "flac",
            "-compression_level",
            quality,
        ]

    raise RuntimeError(
        f"Unsupported conversion format: {conversion.spec}"
    )


def conversion_extension(
    conversion: ConversionFormat,
) -> str:
    """Return the filename extension for a conversion target."""
    extensions = {
        "mp3": ".mp3",
        "aac": ".m4a",
        "m4a": ".m4a",
        "opus": ".opus",
        "flac": ".flac",
    }

    try:
        return extensions[
            conversion.container.lower()
        ]
    except KeyError as exc:
        raise RuntimeError(
            f"Unsupported conversion format: {conversion.spec}"
        ) from exc


def ffmpeg_command(
    url: str,
    output: Path,
    token: str,
    conversion: ConversionFormat,
    copy: bool,
) -> list[str]:
    """Build a retry-safe FFmpeg command."""
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

    if copy:
        command.extend(
            [
                "-c:a",
                "copy",
                "-f",
                conversion.container,
            ]
        )
    else:
        command.extend(
            ffmpeg_quality_args(
                conversion
            )
        )

        command.extend(
            [
                "-f",
                conversion.container,
            ]
        )

    command.append(str(output))

    return command


def run_ffmpeg(
    url: str,
    destination: Path,
    token: str,
    timeout: int,
    conversion: ConversionFormat,
    copy: bool,
    expected_duration_ms: int,
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
        part_path.unlink(
            missing_ok=True
        )
    except OSError:
        pass

    command = ffmpeg_command(
        url=url,
        output=part_path,
        token=token,
        conversion=conversion,
        copy=copy,
    )

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(
                60,
                timeout,
            ) * 30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        part_path.unlink(missing_ok=True)

        return (
            False,
            0,
            "ffmpeg timed out",
        )
    except OSError as exc:
        return (
            False,
            0,
            str(exc),
        )

    if completed.returncode != 0:
        error = (
            completed.stderr.strip()
            or (
                "ffmpeg exited with code "
                f"{completed.returncode}"
            )
        )

        part_path.unlink(missing_ok=True)

        return (
            False,
            0,
            error[-1600:],
        )

    try:
        size = part_path.stat().st_size
    except OSError:
        size = 0

    if size < 1024:
        part_path.unlink(missing_ok=True)

        return (
            False,
            0,
            "ffmpeg produced an empty or incomplete file",
        )

    valid, error = file_is_valid(
        part_path,
        expected_duration_ms,
    )

    if not valid:
        part_path.unlink(missing_ok=True)

        return (
            False,
            0,
            error,
        )

    try:
        os.replace(
            part_path,
            destination,
        )
    except OSError as exc:
        part_path.unlink(missing_ok=True)

        return (
            False,
            0,
            str(exc),
        )

    return (
        True,
        size,
        "",
    )


def copy_source(
    url: str,
    destination: Path,
    token: str,
    timeout: int,
    expected_duration_ms: int,
) -> tuple[bool, int, str]:
    """Copy an already-supported source into an atomic destination."""
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    part_path = destination.with_name(
        destination.name + ".part"
    )

    part_path.unlink(missing_ok=True)

    request = build_request(
        url,
        token=token,
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            with part_path.open(
                "wb"
            ) as output:
                while True:
                    chunk = response.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    output.write(chunk)

    except (
        OSError,
        urllib.error.URLError,
        urllib.error.HTTPError,
    ) as exc:
        part_path.unlink(missing_ok=True)

        return (
            False,
            0,
            str(exc),
        )

    try:
        size = part_path.stat().st_size
    except OSError:
        size = 0

    if size < 1024:
        part_path.unlink(missing_ok=True)

        return (
            False,
            0,
            "source transfer was empty or incomplete",
        )

    valid, error = file_is_valid(
        part_path,
        expected_duration_ms,
    )

    if not valid:
        part_path.unlink(missing_ok=True)

        return (
            False,
            0,
            error,
        )

    try:
        os.replace(
            part_path,
            destination,
        )
    except OSError as exc:
        part_path.unlink(missing_ok=True)

        return (
            False,
            0,
            str(exc),
        )

    return (
        True,
        size,
        "",
    )


def download_track(
    job: DownloadJob,
    server: PlexServer,
    conversion: ConversionFormat,
    retries: int,
    retry_delay: float,
    timeout: int,
) -> DownloadResult:
    """Download one track, retrying invalid transfers."""
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

    copy = source_matches_conversion(
        job.track,
        conversion,
    )

    last_error = "unknown error"

    for attempt in range(
        1,
        retries + 1,
    ):
        if copy:
            success, size, error = copy_source(
                url=job.track.media_url,
                destination=destination,
                token=server.token,
                timeout=timeout,
                expected_duration_ms=job.track.duration_ms,
            )
        else:
            success, size, error = run_ffmpeg(
                url=job.track.media_url,
                destination=destination,
                token=server.token,
                timeout=timeout,
                conversion=conversion,
                copy=False,
                expected_duration_ms=job.track.duration_ms,
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
        "?"
        if result.job.total <= 0
        else str(result.job.total)
    )

    print(
        f"  [{result.job.index:>4}/"
        f"{total:<4}] "
        f"{status} "
        f"{compact(label, 55):<55} "
        f"{human_size(result.bytes_written):>10} "
        f"{human_rate(rate):>12} "
        f"{human_size(remaining):>10}"
    )

    if result.skipped:
        print(
            "       ↳ already downloaded"
        )

    if not result.success:
        print(
            f"       ! {compact(' '.join(result.error.split()), 140)}"
        )


def create_jobs(
    tracks: list[Track],
    playlist_name: str,
    output_root: Path,
    directory_limit: int,
    extension: str,
) -> list[DownloadJob]:
    """Create deterministic download jobs."""
    jobs = []

    for index, track in enumerate(
        tracks,
        1,
    ):
        jobs.append(
            DownloadJob(
                index=index,
                total=len(tracks),
                track=track,
                destination=build_output_path(
                    root=output_root,
                    playlist_name=playlist_name,
                    position=index,
                    track=track,
                    directory_limit=directory_limit,
                    extension=extension,
                ),
            )
        )

    return jobs


def download_jobs(
    jobs: list[DownloadJob],
    output_root: Path,
    server: PlexServer,
    conversion: ConversionFormat,
    retries: int,
    retry_delay: float,
    timeout: int,
    conversion_workers_count: int,
    queue_size: int,
) -> tuple[int, int, int]:
    """Process a deep queue while limiting simultaneous conversions."""
    downloaded = 0
    failed = 0
    written = 0

    # The executor owns the conversion limit. The submission window is larger
    # so network and USB work can stay queued without spawning more converters.
    pending: set[
        concurrent.futures.Future[DownloadResult]
    ] = set()

    iterator = iter(jobs)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=conversion_workers_count,
        thread_name_prefix="plex",
    ) as executor:
        while True:
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
                        conversion,
                        retries,
                        retry_delay,
                        timeout,
                    )
                )

            if not pending:
                break

            done, pending = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
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


def download_playlist(
    tracks: list[Track],
    playlist_name: str,
    output_root: Path,
    directory_limit: int,
    extension: str,
    server: PlexServer,
    conversion: ConversionFormat,
    retries: int,
    retry_delay: float,
    timeout: int,
    conversion_workers_count: int,
    queue_size: int,
) -> tuple[int, int, int]:
    """Download one playlist."""
    if not tracks:
        return 0, 0, 0

    jobs = create_jobs(
        tracks=tracks,
        playlist_name=playlist_name,
        output_root=output_root,
        directory_limit=directory_limit,
        extension=extension,
    )

    print()
    print(f"  {playlist_name}")
    print(
        f"  {'─' * min(72, len(playlist_name) + 2)}"
    )
    print(f"  Tracks:          {len(jobs):,}")
    print(
        f"  Conversion:      {conversion.spec}"
    )
    print(
        f"  Conversion jobs: {conversion_workers_count}"
    )
    print(
        f"  Queue:           {queue_size}"
    )
    print()

    return download_jobs(
        jobs=jobs,
        output_root=output_root,
        server=server,
        conversion=conversion,
        retries=retries,
        retry_delay=retry_delay,
        timeout=timeout,
        conversion_workers_count=conversion_workers_count,
        queue_size=queue_size,
    )


def download_random_fill(
    tracks: list[Track],
    output_root: Path,
    directory_limit: int,
    extension: str,
    server: PlexServer,
    conversion: ConversionFormat,
    retries: int,
    retry_delay: float,
    timeout: int,
    conversion_workers_count: int,
    queue_size: int,
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
            for path in random_root.rglob("*")
            if path.suffix.lower() == extension.lower()
        )
        if random_root.exists()
        else 0
    )

    print()
    print("  Random")
    print("  ──────")
    print(
        f"  Library tracks:   {len(tracks):,}"
    )
    print(
        f"  New candidates:   {len(candidates):,}"
    )
    print(
        f"  Conversion:       {conversion.spec}"
    )
    print(
        f"  Conversion jobs:  {conversion_workers_count}"
    )
    print(
        f"  Queue:            {queue_size}"
    )
    print(
        "  Fill mode:        until drive capacity"
    )
    print(
        f"  Safety reserve:   {human_size(RANDOM_FILL_RESERVE)}"
    )
    print()

    downloaded = 0
    failed = 0
    written = 0
    position = existing_count + 1
    cursor = 0

    while (
        cursor < len(candidates)
        and random_fill_space_available(output_root)
    ):
        batch_size = min(
            queue_size,
            len(candidates) - cursor,
        )

        batch = candidates[
            cursor : cursor + batch_size
        ]

        jobs = []

        for track in batch:
            jobs.append(
                DownloadJob(
                    index=position,
                    total=0,
                    track=track,
                    destination=build_output_path(
                        root=output_root,
                        playlist_name="Random",
                        position=position,
                        track=track,
                        directory_limit=directory_limit,
                        extension=extension,
                    ),
                )
            )

            position += 1

        cursor += batch_size

        d, f, w = download_jobs(
            jobs=jobs,
            output_root=output_root,
            server=server,
            conversion=conversion,
            retries=retries,
            retry_delay=retry_delay,
            timeout=timeout,
            conversion_workers_count=conversion_workers_count,
            queue_size=queue_size,
        )

        downloaded += d
        failed += f
        written += w

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

    interrupted = False

    try:
        check_ffmpeg()
        check_ffprobe()

        config = load_config()

        conversion_formats = get_conversion_formats(
            config
        )

        conversion = conversion_formats[0]

        conversion_workers_count = conversion_workers(
            config["audio"].get(
                "conversion_threads",
                "auto",
            )
        )

        queue_size = max(
            1,
            int(
                config["download"].get(
                    "queue_size",
                    DEFAULT_QUEUE_SIZE,
                )
            ),
        )

        server = prompt_server(config)

        timeout = int(
            config["plex"].get(
                "timeout",
                30,
            )
        )

        library_key, section_name = select_music_library(
            server,
            timeout,
        )

        print()
        print(
            f"Music library: {section_name}"
        )

        playlists = get_playlists(
            server,
            timeout,
        )

        selected = choose_playlists(
            playlists
        )

        directory_limit = choose_directory_limit()

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

        extension = conversion_extension(
            conversion
        )

        has_random = any(
            title.casefold().strip() == "random"
            for _, title in selected
        )

        print()
        print("Download configuration")
        print()
        print(
            f"  Server:             {server.name}"
        )
        print(
            f"  Address:            {server.base_url}"
        )
        print(
            f"  Conversion:         {conversion.spec}"
        )
        print(
            f"  Conversion threads: {conversion_workers_count}"
        )
        print(
            f"  Queue:              {queue_size}"
        )
        print(
            f"  Retries:            {retries}"
        )
        print(
            "  Directory:          "
            + (
                "unlimited"
                if directory_limit == -1
                else str(directory_limit)
            )
        )
        print(
            f"  Destination:        {output_root}"
        )

        if has_random:
            print(
                "  Random fill:        enabled"
            )

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

        print()

        grand_downloaded = 0
        grand_failed = 0
        grand_written = 0

        # Explicit playlists run first so Random consumes only remaining space.
        for playlist_name, tracks in selected_tracks:
            if not tracks:
                print(
                    f"  Skipping empty download: "
                    f"{playlist_name}"
                )
                continue

            downloaded, failed, written = download_playlist(
                tracks=tracks,
                playlist_name=playlist_name,
                output_root=output_root,
                directory_limit=directory_limit,
                extension=extension,
                server=server,
                conversion=conversion,
                retries=retries,
                retry_delay=retry_delay,
                timeout=timeout,
                conversion_workers_count=conversion_workers_count,
                queue_size=queue_size,
            )

            grand_downloaded += downloaded
            grand_failed += failed
            grand_written += written

        if random_selected:
            downloaded, failed, written = download_random_fill(
                tracks=random_tracks,
                output_root=output_root,
                directory_limit=directory_limit,
                extension=extension,
                server=server,
                conversion=conversion,
                retries=retries,
                retry_delay=retry_delay,
                timeout=timeout,
                conversion_workers_count=conversion_workers_count,
                queue_size=queue_size,
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
        interrupted = True

        # Keep the interruption state visually distinct before replacing it.
        print(
            "\r  Aborting...        ",
            end="",
            flush=True,
        )

        time.sleep(0.15)

        print(
            "\r  Interrupted.       "
        )

        return 130

    except Exception as exc:
        print()
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 1

    finally:
        if interrupted:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
