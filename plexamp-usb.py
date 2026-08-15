#!/usr/bin/env python3
"""Export Plex music to a car-friendly USB filesystem.

The program discovers a reachable Plex Media Server, authenticates only when
local unauthenticated access is unavailable, lets the user select Plex music
playlists, and exports tracks directly below a Downloads directory beside
this script.

Supported source formats are copied directly when they already match the
configured conversion format. Other formats are converted with FFmpeg.

Downloads are deterministic and resumable. Existing complete files are kept;
incomplete, invalid, or mismatched files are replaced and retried.

Directories can be capped at a user-selected number of audio files. A value
of -1 disables the limit.

A playlist named "Random" is treated as a special fill mode. Tracks are
sampled from the complete Plex music library and downloaded until the
destination filesystem reaches its safety threshold.

The existing first-level directories below Downloads are used as the default
playlist selection on each run. Playlist defaults are intentionally not
stored in settings.json.

Configuration is stored in settings.json beside this script.

Requirements:
    - Python 3.10+
    - ffmpeg and ffprobe when conversion is required
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

# Keep enough space for filesystem metadata and the final output operation.
RANDOM_FILL_RESERVE = 128 * 1024 * 1024

# Converted files may differ slightly from Plex's source duration because of
# encoder delay and container timestamps. Large differences indicate a bad
# transfer or incomplete source and must be retried.
DURATION_TOLERANCE_SECONDS = 2.0

DEFAULT_CONFIG = {
    "plex": {
        "host": "",
        "port": 32400,
        "token": "",
        "timeout": 30,
    },
    "audio": {
        "conversion_formats": [
            "mp3:V0",
        ],
        "conversion_threads": "auto",
    },
    "output": {
        "directory": DOWNLOAD_DIR,
        "directory_limit": 255,
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
        """Return the server's base HTTP URL."""
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


def sanitize_filename(
    value: str,
    fallback: str = "Unknown",
    max_bytes: int = 255,
) -> str:
    """Return a deterministic filesystem-safe name."""
    value = unicodedata.normalize(
        "NFC",
        str(value or ""),
    )
    value = value.replace("\x00", "")

    value = re.sub(
        r'[<>:"/\\|?*\x00-\x1f\x7f]',
        "_",
        value,
    )
    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()
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

    return (
        suffix.encode("utf-8")[:max_bytes]
        .decode("utf-8", "ignore")
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

    prefix_part = (
        f"{prefix} - {artist} - {album} - "
    )
    extension = ".mp3"

    available = (
        255
        - len(
            (prefix_part + extension).encode(
                "utf-8"
            )
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
                (prefix_part + extension).encode(
                    "utf-8"
                )
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


def save_config(path: Path, config: dict) -> None:
    """Write configuration atomically."""
    temporary = path.with_name(
        path.name + ".part"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            config,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    os.replace(
        temporary,
        path,
    )


def load_config() -> dict:
    """Load settings.json and create it from defaults when absent."""
    path = (
        Path(__file__).resolve().parent
        / CONFIG_JSON
    )

    if not path.exists():
        config = deep_merge(
            {},
            DEFAULT_CONFIG,
        )
        save_config(
            path,
            config,
        )
        print(f"  Created {path}")
        return config

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
    """Test Plex connectivity and return identity."""
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

        deadline = (
            time.monotonic()
            + timeout
        )

        while time.monotonic() < deadline:
            try:
                data, address = sock.recvfrom(
                    8192
                )
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

                headers[
                    key.strip().lower()
                ] = value.strip()

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
    """Select a Plex server, preferring local access."""
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
            "  Configured server unavailable: "
            f"{configured.base_url}"
        )

    print(
        "  Discovering local Plex servers..."
    )

    servers = discover_gdm_servers()

    if not servers:
        print()
        print(
            "  No local Plex server was discovered."
        )
        print()
        print(
            "  Enter the Plex server address manually,"
        )
        print(
            "  or set plex.host in settings.json."
        )
        print()

        address = input(
            "  Plex server address: "
        ).strip()

        if not address:
            raise RuntimeError(
                "No Plex server address supplied."
            )

        parsed = urllib.parse.urlparse(
            address
        )

        if parsed.scheme:
            protocol = parsed.scheme
            host = parsed.hostname or address
            port = (
                parsed.port
                or configured_port
            )
        else:
            protocol = "http"

            if ":" in address:
                possible_host, possible_port = (
                    address.rsplit(":", 1)
                )

                try:
                    port = int(
                        possible_port
                    )
                    host = possible_host
                except ValueError:
                    host = address
                    port = configured_port
            else:
                host = address
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
                    server = servers[
                        index - 1
                    ]
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
    print(
        "  Local access requires authentication."
    )

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
                return libraries[
                    index - 1
                ]
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


def default_playlist_names(
    output_root: Path,
) -> set[str]:
    """Use existing top-level download directories as defaults."""
    if not output_root.is_dir():
        return set()

    return {
        path.name.casefold()
        for path in output_root.iterdir()
        if path.is_dir()
    }


def choose_playlists(
    playlists: list[tuple[str, str, int, str]],
    output_root: Path,
) -> list[tuple[str, str]]:
    """Select one or more Plex playlists.

    Existing Downloads subdirectories are selected by default. The filesystem
    remains the persistent state instead of duplicating that state in JSON.
    """
    if not playlists:
        raise RuntimeError(
            "No Plex music playlists were found."
        )

    defaults = default_playlist_names(
        output_root
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
        marker = (
            "*"
            if playlist_directory(title).casefold()
            in defaults
            else " "
        )

        print(
            f"  {marker} {index:>2}. "
            f"{title} "
            f"({count:,} tracks, {duration})"
        )

    print("     X. Random Fill Mode")
    print("     A. All")
    print("     Multiple selections: 1,3,5,X")

    default_indices = [
        index
        for index, (
            _,
            title,
            _,
            _,
        ) in enumerate(
            playlists,
            1,
        )
        if playlist_directory(title).casefold()
        in defaults
    ]

    default_random = (
        "random" in defaults
    )

    if default_indices or default_random:
        selected_text = ",".join(
            str(index)
            for index in default_indices
        )

        if default_random:
            selected_text += (
                ","
                if selected_text
                else ""
            ) + "X"

        prompt = (
            f"\nSelect downloads "
            f"[{selected_text}]: "
        )
    else:
        prompt = "\nSelect downloads: "

    while True:
        answer = input(prompt).strip()

        if not answer and (
            default_indices
            or default_random
        ):
            selected = [
                (
                    playlists[index - 1][0],
                    playlists[index - 1][1],
                )
                for index in default_indices
            ]

            if default_random:
                selected.append(
                    ("Random", "Random")
                )

            return selected

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

                if not part:
                    continue

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


def track_from_xml(
    item: ET.Element,
    server: PlexServer,
    playlist_id: str,
) -> Track | None:
    """Build a Track from a Plex Track element."""
    media = item.find("Media")

    if media is None:
        return None

    part = media.find("Part")

    if part is None:
        return None

    key = part.attrib.get(
        "key",
        "",
    )

    if not key:
        return None

    media_url = urllib.parse.urljoin(
        server.base_url,
        key,
    )

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
        media_url=media_url,
        source_size=source_size,
        playlist_id=playlist_id,
        container=part.attrib.get(
            "container",
            media.attrib.get(
                "container",
                "",
            ),
        ),
        audio_codec=part.attrib.get(
            "audioCodec",
            media.attrib.get(
                "audioCodec",
                "",
            ),
        ),
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

    for item in root.findall("Track"):
        track = track_from_xml(
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
    """Fetch all playable audio tracks from a Plex music library."""
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
        track = track_from_xml(
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
    """Prompt for the maximum number of audio files per directory."""
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

        print(
            "  Enter a positive number or -1."
        )


def playlist_directory(
    playlist_name: str,
) -> str:
    """Return a stable playlist directory name."""
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
        / playlist_directory(
            playlist_name
        )
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
        )
    )


def conversion_spec(
    config: dict,
) -> tuple[str, str]:
    """Return the first configured conversion format."""
    formats = config["audio"].get(
        "conversion_formats",
        [],
    )

    if not isinstance(formats, list) or not formats:
        raise RuntimeError(
            "audio.conversion_formats must contain at least one format."
        )

    spec = str(formats[0]).strip()

    if not spec:
        raise RuntimeError(
            "audio.conversion_formats contains an empty format."
        )

    parts = spec.split(
        ":",
        1,
    )

    output_format = parts[0].strip().lower()
    quality = (
        parts[1].strip()
        if len(parts) == 2
        else ""
    )

    if not re.fullmatch(
        r"[a-z0-9._+-]+",
        output_format,
    ):
        raise RuntimeError(
            f"Invalid conversion format: {spec}"
        )

    return output_format, quality


def conversion_threads(config: dict) -> int:
    """Return the configured maximum number of concurrent jobs."""
    value = config["audio"].get(
        "conversion_threads",
        "auto",
    )

    if (
        isinstance(value, str)
        and value.strip().lower() == "auto"
    ):
        return os.cpu_count() or 1

    try:
        workers = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "audio.conversion_threads must be "
            '"auto" or a positive integer.'
        ) from exc

    if workers <= 0:
        raise RuntimeError(
            "audio.conversion_threads must be positive."
        )

    return workers


def source_matches_output(
    track: Track,
    output_format: str,
) -> bool:
    """Return whether the source can be copied without conversion."""
    container = (
        track.container or ""
    ).casefold()

    codec = (
        track.audio_codec or ""
    ).casefold()

    return (
        container == output_format
        or codec == output_format
        or (
            output_format == "mp3"
            and track.media_url.casefold().split(
                "?",
                1,
            )[0].endswith(".mp3")
        )
    )


def check_program(
    program: str,
) -> bool:
    """Return whether an executable is available in PATH."""
    try:
        completed = subprocess.run(
            [program, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False

    return completed.returncode == 0


def check_conversion_tools(
    config: dict,
) -> None:
    """Verify FFmpeg only when conversion is actually required."""
    formats = config["audio"].get(
        "conversion_formats",
        [],
    )

    if not formats:
        raise RuntimeError(
            "audio.conversion_formats must not be empty."
        )

    # Direct copies do not require FFmpeg. The tools are checked lazily once
    # a track requiring conversion is encountered.
    return


def ensure_ffmpeg() -> None:
    """Verify FFmpeg and FFprobe are available."""
    if not check_program("ffmpeg"):
        raise RuntimeError(
            "ffmpeg is required for conversion but was not found in PATH."
        )

    if not check_program("ffprobe"):
        raise RuntimeError(
            "ffprobe is required to validate converted durations "
            "but was not found in PATH."
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


def track_identity(track: Track) -> str:
    """Return a normalized identity for duplicate detection."""
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


def exported_track_keys(
    output_root: Path,
) -> set[str]:
    """Build conservative identities from exported filenames."""
    keys: set[str] = set()

    if not output_root.exists():
        return keys

    for path in output_root.rglob("*.mp3"):
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
        if track_identity(track)
        not in existing
    ]

    random.SystemRandom().shuffle(
        candidates
    )

    return candidates


def parse_quality(
    output_format: str,
    quality: str,
) -> list[str]:
    """Translate the universal conversion quality into FFmpeg options."""
    if not quality:
        return []

    normalized = quality.upper()

    if output_format == "mp3":
        if re.fullmatch(
            r"V[0-9]",
            normalized,
        ):
            return [
                "-q:a",
                normalized[1:],
            ]

        if re.fullmatch(
            r"[0-9]+K",
            normalized,
        ):
            return [
                "-b:a",
                normalized.lower(),
            ]

        raise RuntimeError(
            "MP3 conversion quality must be "
            "V0-V9 or a bitrate such as 192k."
        )

    if re.fullmatch(
        r"[0-9]+K",
        normalized,
    ):
        return [
            "-b:a",
            normalized.lower(),
        ]

    return [
        "-q:a",
        quality,
    ]


def ffmpeg_command(
    track: Track,
    output: Path,
    token: str,
    output_format: str,
    quality: str,
) -> list[str]:
    """Build an FFmpeg conversion command."""
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-threads",
        "1",
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
            track.media_url,
            "-vn",
            "-map",
            "0:a:0",
            "-map_metadata",
            "0",
        ]
    )

    # Embedded artwork is represented as a video stream by FFmpeg. Preserve
    # it when the selected output container supports attached pictures.
    if output_format == "mp3":
        command.extend(
            [
                "-map",
                "0:v?",
                "-c:v",
                "copy",
                "-id3v2_version",
                "3",
            ]
        )

    command.extend(
        [
            "-c:a",
            (
                "libmp3lame"
                if output_format == "mp3"
                else output_format
            ),
        ]
    )

    command.extend(
        parse_quality(
            output_format,
            quality,
        )
    )

    command.extend(
        [
            "-f",
            output_format,
            str(output),
        ]
    )

    return command


def probe_duration(
    path: Path,
) -> float | None:
    """Return the duration of a completed media file."""
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-hide_banner",
                "-loglevel",
                "error",
                "-show_entries",
                "format=duration",
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


def duration_matches(
    path: Path,
    expected_ms: int,
) -> tuple[bool, str]:
    """Validate output duration against Plex metadata."""
    if expected_ms <= 0:
        return True, ""

    actual = probe_duration(path)

    if actual is None:
        return (
            False,
            "unable to read output duration",
        )

    expected = expected_ms / 1000

    if abs(actual - expected) > DURATION_TOLERANCE_SECONDS:
        return (
            False,
            "duration mismatch: "
            f"got {human_duration(int(actual * 1000))}, "
            f"expected {human_duration(expected_ms)}",
        )

    return True, ""


def direct_file_valid(
    path: Path,
    track: Track,
) -> tuple[bool, str]:
    """Validate a directly copied source by its expected byte count."""
    try:
        size = path.stat().st_size
    except OSError:
        return False, "file is missing"

    if size < 1024:
        return False, "file is too small"

    if (
        track.source_size > 0
        and size != track.source_size
    ):
        return (
            False,
            "size mismatch: "
            f"got {human_size(size)}, "
            f"expected {human_size(track.source_size)}",
        )

    return True, ""


def copy_track(
    track: Track,
    destination: Path,
    token: str,
    timeout: int,
) -> tuple[bool, int, str]:
    """Copy an already-supported source without transcoding."""
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

    request = build_request(
        track.media_url,
        token=token,
    )

    written = 0

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            header_size = response.headers.get(
                "Content-Length"
            )

            try:
                content_length = (
                    int(header_size)
                    if header_size
                    else 0
                )
            except ValueError:
                content_length = 0

            if (
                track.source_size > 0
                and content_length > 0
                and content_length != track.source_size
            ):
                return (
                    False,
                    0,
                    "HTTP size mismatch: "
                    f"got {content_length} bytes, "
                    f"expected {track.source_size}",
                )

            with part_path.open(
                "wb"
            ) as handle:
                while True:
                    chunk = response.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    handle.write(chunk)
                    written += len(chunk)

        expected = (
            track.source_size
            or content_length
        )

        if expected > 0 and written != expected:
            raise IOError(
                "incomplete transfer: "
                f"got {written} bytes, "
                f"expected {expected}"
            )

        if written < 1024:
            raise IOError(
                "transfer produced an empty or incomplete file"
            )

        os.replace(
            part_path,
            destination,
        )

        return True, written, ""

    except (
        OSError,
        urllib.error.URLError,
        urllib.error.HTTPError,
    ) as exc:
        try:
            part_path.unlink(
                missing_ok=True
            )
        except OSError:
            pass

        return False, 0, str(exc)


def convert_track(
    track: Track,
    destination: Path,
    token: str,
    timeout: int,
    output_format: str,
    quality: str,
) -> tuple[bool, int, str]:
    """Convert one track into an atomic temporary output."""
    ensure_ffmpeg()

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
        track=track,
        output=part_path,
        token=token,
        output_format=output_format,
        quality=quality,
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
        try:
            part_path.unlink(
                missing_ok=True
            )
        except OSError:
            pass

        return (
            False,
            0,
            "ffmpeg timed out",
        )
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
            part_path.unlink(
                missing_ok=True
            )
        except OSError:
            pass

        return False, 0, error[-1600:]

    try:
        size = part_path.stat().st_size
    except OSError:
        size = 0

    if size < 1024:
        try:
            part_path.unlink(
                missing_ok=True
            )
        except OSError:
            pass

        return (
            False,
            0,
            "ffmpeg produced an empty or incomplete file",
        )

    valid, error = duration_matches(
        part_path,
        track.duration_ms,
    )

    if not valid:
        try:
            part_path.unlink(
                missing_ok=True
            )
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
            part_path.unlink(
                missing_ok=True
            )
        except OSError:
            pass

        return False, 0, str(exc)

    return True, size, ""


def file_is_valid(
    path: Path,
    track: Track,
    output_format: str,
) -> tuple[bool, str]:
    """Validate an existing completed output."""
    if not path.is_file():
        return False, "file is missing"

    if source_matches_output(
        track,
        output_format,
    ):
        return direct_file_valid(
            path,
            track,
        )

    return duration_matches(
        path,
        track.duration_ms,
    )


def download_track(
    job: DownloadJob,
    server: PlexServer,
    retries: int,
    retry_delay: float,
    timeout: int,
    output_format: str,
    quality: str,
) -> DownloadResult:
    """Download one track with deterministic retry semantics."""
    destination = job.destination

    valid, _ = file_is_valid(
        destination,
        job.track,
        output_format,
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

    direct_copy = source_matches_output(
        job.track,
        output_format,
    )

    for attempt in range(
        1,
        retries + 1,
    ):
        if direct_copy:
            success, size, error = copy_track(
                track=job.track,
                destination=destination,
                token=server.token,
                timeout=timeout,
            )
        else:
            success, size, error = convert_track(
                track=job.track,
                destination=destination,
                token=server.token,
                timeout=timeout,
                output_format=output_format,
                quality=quality,
            )

        if success:
            return DownloadResult(
                job=job,
                success=True,
                skipped=False,
                bytes_written=size,
                elapsed=(
                    time.monotonic()
                    - started
                ),
                attempts=attempt,
            )

        last_error = error

        if attempt >= retries:
            break

        delay = min(
            30.0,
            retry_delay
            * (2 ** (attempt - 1)),
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
        elapsed=(
            time.monotonic()
            - started
        ),
        attempts=retries,
        error=last_error,
    )


def print_result(
    result: DownloadResult,
    output_root: Path,
) -> None:
    """Print one compact download result."""
    status = (
        "✓"
        if result.success
        else "✗"
    )

    rate = (
        result.bytes_written
        / result.elapsed
        if result.elapsed > 0
        else 0
    )

    remaining = free_space(
        output_root
    )

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
        print(
            "       ↳ already downloaded"
        )

    if not result.success:
        error = compact(
            " ".join(
                result.error.split()
            ),
            140,
        )

        print(
            f"       ! {error}"
        )


def create_jobs(
    tracks: list[Track],
    playlist_name: str,
    output_root: Path,
    directory_limit: int,
) -> list[DownloadJob]:
    """Create deterministic download jobs."""
    return [
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
            ),
        )
        for index, track in enumerate(
            tracks,
            1,
        )
    ]


def download_jobs(
    jobs: list[DownloadJob],
    output_root: Path,
    server: PlexServer,
    retries: int,
    retry_delay: float,
    timeout: int,
    workers: int,
    output_format: str,
    quality: str,
) -> tuple[int, int, int]:
    """Process a deep queue with bounded concurrent workers."""
    downloaded = 0
    failed = 0
    written = 0

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="plex",
    )

    futures = [
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
        for job in jobs
    ]

    try:
        for future in concurrent.futures.as_completed(
            futures
        ):
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

    except KeyboardInterrupt:
        print(
            "\r  Aborting...",
            end="",
            flush=True,
        )

        for future in futures:
            future.cancel()

        executor.shutdown(
            wait=True,
            cancel_futures=True,
        )

        print(
            "\r  Interrupted.                    "
        )
        raise

    else:
        executor.shutdown(
            wait=True,
        )

    return downloaded, failed, written


def download_playlist(
    tracks: list[Track],
    playlist_name: str,
    output_root: Path,
    directory_limit: int,
    server: PlexServer,
    retries: int,
    retry_delay: float,
    timeout: int,
    workers: int,
    output_format: str,
    quality: str,
) -> tuple[int, int, int]:
    """Download all tracks from one playlist."""
    if not tracks:
        return 0, 0, 0

    jobs = create_jobs(
        tracks=tracks,
        playlist_name=playlist_name,
        output_root=output_root,
        directory_limit=directory_limit,
    )

    print()
    print(f"  {playlist_name}")
    print(
        f"  {'─' * min(72, len(playlist_name) + 2)}"
    )
    print(f"  Tracks:     {len(jobs):,}")
    print(f"  Parallel:   {workers}")
    print(
        f"  Conversion: {output_format}"
        + (
            f":{quality}"
            if quality
            else ""
        )
    )
    print()

    return download_jobs(
        jobs=jobs,
        output_root=output_root,
        server=server,
        retries=retries,
        retry_delay=retry_delay,
        timeout=timeout,
        workers=workers,
        output_format=output_format,
        quality=quality,
    )


def download_random_fill(
    tracks: list[Track],
    output_root: Path,
    directory_limit: int,
    server: PlexServer,
    retries: int,
    retry_delay: float,
    timeout: int,
    workers: int,
    output_format: str,
    quality: str,
) -> tuple[int, int, int]:
    """Fill available USB capacity with random new music."""
    candidates = select_random_tracks(
        tracks,
        output_root,
    )

    if not candidates:
        print()
        print(
            "  Random: no new tracks remain."
        )
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

    position = existing_count + 1
    cursor = 0
    downloaded = 0
    failed = 0
    written = 0

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
        f"  Parallel:         {workers}"
    )
    print(
        f"  Conversion:       {output_format}"
        + (
            f":{quality}"
            if quality
            else ""
        )
    )
    print(
        "  Fill mode:        until drive capacity"
    )
    print(
        f"  Safety reserve:   "
        f"{human_size(RANDOM_FILL_RESERVE)}"
    )
    print()

    # Queue deeply enough to keep the configured workers busy. Capacity is
    # checked between batches so the filesystem is never intentionally filled
    # far beyond its safety reserve.
    while (
        cursor < len(candidates)
        and random_fill_space_available(
            output_root
        )
    ):
        batch = candidates[
            cursor : cursor + workers
        ]
        cursor += len(batch)

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
                    ),
                )
            )
            position += 1

        try:
            d, f, w = download_jobs(
                jobs=jobs,
                output_root=output_root,
                server=server,
                retries=retries,
                retry_delay=retry_delay,
                timeout=timeout,
                workers=workers,
                output_format=output_format,
                quality=quality,
            )
        except KeyboardInterrupt:
            raise

        downloaded += d
        failed += f
        written += w

        if not random_fill_space_available(
            output_root
        ):
            break

    remaining = free_space(
        output_root
    )

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
        config = load_config()

        check_conversion_tools(
            config
        )

        server = prompt_server(
            config
        )

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
        print(
            f"Music library: {section_name}"
        )

        playlists = get_playlists(
            server,
            timeout,
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

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        selected = choose_playlists(
            playlists,
            output_root,
        )

        configured_limit = config["output"].get(
            "directory_limit"
        )

        if configured_limit is not None:
            try:
                directory_limit = int(
                    configured_limit
                )
            except (TypeError, ValueError):
                directory_limit = (
                    choose_directory_limit()
                )
        else:
            directory_limit = (
                choose_directory_limit()
            )

        if directory_limit == 0 or directory_limit < -1:
            directory_limit = (
                choose_directory_limit()
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

        workers = conversion_threads(
            config
        )

        output_format, quality = (
            conversion_spec(
                config
            )
        )

        has_random = any(
            title.casefold().strip()
            == "random"
            for _, title in selected
        )

        print()
        print(
            "Download configuration"
        )
        print()
        print(
            f"  Server:       {server.name}"
        )
        print(
            f"  Address:      {server.base_url}"
        )
        print(
            f"  Conversion:   {output_format}"
            + (
                f":{quality}"
                if quality
                else ""
            )
        )
        print(
            f"  Parallel:     {workers} workers"
        )
        print(
            f"  Retries:      {retries}"
        )
        print(
            "  Directory:    "
            + (
                "unlimited"
                if directory_limit == -1
                else str(directory_limit)
            )
        )
        print(
            f"  Destination:  {output_root}"
        )

        if has_random:
            print(
                "  Random fill:  enabled"
            )

        print()

        selected_tracks: list[
            tuple[str, list[Track]]
        ] = []

        random_selected = False

        print(
            "Collecting music..."
        )
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

            d, f, w = download_playlist(
                tracks=tracks,
                playlist_name=playlist_name,
                output_root=output_root,
                directory_limit=directory_limit,
                server=server,
                retries=retries,
                retry_delay=retry_delay,
                timeout=timeout,
                workers=workers,
                output_format=output_format,
                quality=quality,
            )

            grand_downloaded += d
            grand_failed += f
            grand_written += w

        if random_selected:
            d, f, w = download_random_fill(
                tracks=random_tracks,
                output_root=output_root,
                directory_limit=directory_limit,
                server=server,
                retries=retries,
                retry_delay=retry_delay,
                timeout=timeout,
                workers=workers,
                output_format=output_format,
                quality=quality,
            )

            grand_downloaded += d
            grand_failed += f
            grand_written += w

        remaining = free_space(
            output_root
        )

        print()
        print("─" * 78)
        print(
            "  Music download complete"
        )
        print("─" * 78)
        print(
            f"  Completed               "
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
            print(
                "  ! Some downloads failed."
            )
            print(
                "    Run again to retry them."
            )
            print()
            return 2

        print(
            "  ✓ All downloads completed successfully."
        )
        print()

        return 0

    except KeyboardInterrupt:
        # Worker pools handle the visible abort message so this path remains
        # quiet if Ctrl-C arrives while the main workflow is waiting.
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
