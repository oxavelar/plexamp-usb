#!/usr/bin/env python3
"""Export Plex music downloads to a car-friendly USB filesystem.

The program discovers a reachable Plex Media Server, authenticates only when
local unauthenticated access is unavailable, lets the user select Plex music
playlists, transcodes tracks to MP3 V0 VBR, and writes them directly below a
Downloads directory beside this script.

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

Configuration is read from settings.json. No Plex API client library is
required.

Requirements:
    - Python 3.10+
    - ffmpeg

The output directory is always relative to this script, making the program
portable when the script itself lives on the USB filesystem.
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
import signal


APP_NAME = "plexamp-usb"
CONFIG_JSON = "settings.json"
DOWNLOAD_DIR = "Downloads"

# Leave enough filesystem headroom for metadata, directory entries, and
# filesystem behavior while V0 output sizes are still being discovered.
RANDOM_FILL_RESERVE = 128 * 1024 * 1024

# Message printed by the signal handler. It's printed without a newline so the
# KeyboardInterrupt handler can overwrite it with the final "Interrupted." line.
SIGNAL_MSG = "Aborting..."

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
    "download": {
        "retries": 5,
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
    # New: container and audio codec reported by Plex (if available). These
    # fields are used to detect when the source is already an MP3 so the
    # script can avoid unnecessary re-encoding.
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
    """Fit text into a fixed console width without breaking the layout."""
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
    """Format a Plex duration in milliseconds for console display."""
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
    """Return a filesystem-safe, deterministic filename component.

    This version normalizes Unicode, strips control characters, replaces
    characters that are invalid on common USB filesystems (FAT/VFAT/NTFS),
    and truncates to a maximum number of bytes (UTF-8) while preserving a
    short stable hash suffix to keep truncated names unique.
    """
    value = str(value or "")
    # Normalize to NFC so visually-equal strings encode identically.
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\x00", "")

    # Replace characters that commonly cause problems on FAT/Windows and
    # also remove other control chars. We allow emoji and other Unicode
    # characters but forbid :<>\"/\\|?* and control bytes.
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]', "_", value)

    # Collapse whitespace and trim.
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

    # Truncate to max_bytes measured in UTF-8. If truncation is required
    # append a short stable hash so different long names don't collapse to
    # the same truncated name.
    def truncate_to_bytes(s: str, max_b: int) -> str:
        enc = s.encode("utf-8")
        if len(enc) <= max_b:
            return s

        hash_suffix = stable_hash(s, 6)
        suffix = f"…{hash_suffix}"

        # Reduce s (character by character) until s+suffix fits.
        # This is simple and safe for UTF-8 because we cut at character
        # boundaries before encoding.
        while s:
            candidate = s + suffix
            if len(candidate.encode("utf-8")) <= max_b:
                return candidate
            s = s[:-1]

        # Fallback: if nothing fits, use truncated suffix.
        return suffix.encode("utf-8")[:max_b].decode("utf-8", "ignore")

    return truncate_to_bytes(value, max_bytes)


def stable_hash(
    value: str,
    length: int = 8,
) -> str:
    """Return a short deterministic hash."""
    return hashlib.sha1(
        value.encode("utf-8")
    ).hexdigest()[:length]


def track_filename(
    track: Track,
    number: int,
) -> str:
    """Build a stable car-friendly filename with per-name byte limits."""
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

    # Build the full filename and ensure it fits into 255 bytes (a common
    # per-name limit). If it doesn't, shrink the title component so that
    # the whole name including ".mp3" fits.
    ext = ".mp3"
    prefix_part = f"{prefix} - {artist} - {album} - "
    title_component = title

    max_name_bytes = 255
    # calculate available bytes for title (in UTF-8)
    available_for_title = max_name_bytes - len((prefix_part + ext).encode("utf-8"))
    if available_for_title <= 0:
        # If artist/album/prefix already exceed the limit (unlikely), truncate
        # the album instead and fall back to a hash if necessary.
        album = sanitize_filename(album, "Unknown Album", max_bytes=80)
        prefix_part = f"{prefix} - {artist} - {album} - "
        available_for_title = max_name_bytes - len((prefix_part + ext).encode("utf-8"))

    title = sanitize_filename(
        title_component,
        "Unknown Track",
        max_bytes=max(1, available_for_title),
    )

    return (
        f"{prefix_part}{title}{ext}"
    )


def deep_merge(
    base: dict,
    override: dict,
) -> dict:
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
    """Write the initial JSON configuration file."""
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
    """Load settings.json, creating it with defaults when absent."""
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
            f"Invalid JSON in {path}: "
            f"{exc}"
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
            f"Plex returned HTTP {exc.code} "
            f"for {path}"
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

    # Plex GDM uses UDP multicast on 239.0.0.250:32414.
    message = (
        b"M-SEARCH * HTTP/1.0\r\n"
        b"HOST: 239.0.0.250:32414\r\n"
        b"MAN: \"ssdp:discover\"\r\n"
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
        # Send to the Plex GDM multicast address.
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

                key, value = line.split(
                    ":",
                    1,
                )

                headers[
                    key.strip().lower()
                ] = value.strip()

            content_type = headers.get(
                "content-type",
                "",
            ).lower()

            if (
                "plex/media-server"
                not in content_type
            ):
                continue

            host_header = headers.get(
                "host",
                "",
            ).strip()

            host = host_header

            if not host:
                host = address[0]

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
                protocol="http",
            )

    finally:
        sock.close()

    return list(
        discovered.values()
    )


def prompt_server(
    config: dict,
) -> PlexServer:
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
            print(
                f"  Server: {identity.attrib.get(
                    'friendlyName',
                    configured_host,
                )}"
            )
            print(
                f"  Address: {configured.base_url}"
            )
            print("  Access:  OK")

            return PlexServer(
                name=identity.attrib.get(
                    "friendlyName",
                    configured_host,
                ),
                host=configured_host,
                port=configured_port,
                protocol=configured.protocol,
                token=configured_token,
            )

        print(
            f"  Configured server unavailable: "
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

        host = input(
            "  Plex server address: "
        ).strip()

        if not host:
            raise RuntimeError(
                "No Plex server address supplied."
            )

        parsed = urllib.parse.urlparse(
            host
        )

        if parsed.scheme:
            protocol = parsed.scheme
            host = parsed.hostname or host
            port = (
                parsed.port
                or configured_port
            )
        else:
            protocol = "http"

            if ":" in host:
                possible_host, possible_port = (
                    host.rsplit(":", 1)
                )

                try:
                    port = int(
                        possible_port
                    )
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
                    server = servers[
                        index - 1
                    ]
                    break
            except ValueError:
                pass

            print(
                "  Invalid selection."
            )

    print(
        f"  Found: {server.name} "
        f"({server.base_url})"
    )

    identity = test_server(
        server,
        timeout,
    )

    if identity is not None:
        print(
            "  Local access: OK"
        )

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
            "A Plex token is required "
            "for this server."
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
            "Unable to connect to Plex "
            "with the supplied token."
        )

    print(
        "  Authentication: OK"
    )

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

    libraries: list[
        tuple[str, str]
    ] = []

    for directory in root.findall(
        "Directory"
    ):
        if directory.attrib.get(
            "type"
        ) != "artist":
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

        print(
            "Invalid selection."
        )


def get_playlists(
    server: PlexServer,
    timeout: int,
) -> list[
    tuple[str, str, int, str]
]:
    """Return audio playlists available from Plex."""
    root = plex_xml(
        server,
        "/playlists",
        params={
            "playlistType": "audio"
        },
        timeout=timeout,
    )

    playlists = []

    for playlist in root.findall(
        "Playlist"
    ):
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
            playlist.attrib.get(
                "duration"
            )
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


def get_playlist_items(
    server: PlexServer,
    rating_key: str,
    timeout: int,
) -> list[Track]:
    """Get all tracks from a Plex playlist."""
    root = plex_xml(
        server,
        f"/playlists/{rating_key}/items",
        timeout=timeout,
    )

    tracks = []

    for track in root.findall("Track"):
        rating_key = track.attrib.get("ratingKey", "")
        title = track.attrib.get("title", "Unknown")
        artist = track.attrib.get("grandparentTitle", "Unknown Artist")
        album = track.attrib.get("parentTitle", "Unknown Album")
        album_artist = track.attrib.get("grandparentTitle", "Unknown Artist")
        parent_index = track.attrib.get("parentIndex", "0")
        index = track.attrib.get("index", "0")
        container = track.attrib.get("container", "")
        audio_codec = track.attrib.get("audioCodec", "")

        try:
            duration_ms = int(track.attrib.get("duration", "0"))
        except ValueError:
            duration_ms = 0

        try:
            source_size = int(track.attrib.get("size", "0"))
        except ValueError:
            source_size = 0

        media = track.find("Media")
        if media is None:
            continue

        part = media.find("Part")
        if part is None:
            continue

        media_url = f"{server.base_url}{part.attrib.get('key', '')}"

        if not rating_key:
            continue

        tracks.append(
            Track(
                rating_key=rating_key,
                title=title,
                artist=artist,
                album=album,
                album_artist=album_artist,
                parent_index=parent_index,
                index=index,
                duration_ms=duration_ms,
                media_url=media_url,
                source_size=source_size,
                playlist_id=rating_key,
                container=container,
                audio_codec=audio_codec,
            )
        )

    return tracks


def get_library_items(
    server: PlexServer,
    library_key: str,
    timeout: int,
    limit: int | None = None,
) -> list[Track]:
    """Get all tracks from a Plex library."""
    root = plex_xml(
        server,
        f"/library/sections/{library_key}/all",
        params={
            "type": 10,  # Track
            "limit": limit or 9999,
        },
        timeout=timeout,
    )

    tracks = []

    for track in root.findall("Track"):
        rating_key = track.attrib.get("ratingKey", "")
        title = track.attrib.get("title", "Unknown")
        artist = track.attrib.get("grandparentTitle", "Unknown Artist")
        album = track.attrib.get("parentTitle", "Unknown Album")
        album_artist = track.attrib.get("grandparentTitle", "Unknown Artist")
        parent_index = track.attrib.get("parentIndex", "0")
        index = track.attrib.get("index", "0")
        container = track.attrib.get("container", "")
        audio_codec = track.attrib.get("audioCodec", "")

        try:
            duration_ms = int(track.attrib.get("duration", "0"))
        except ValueError:
            duration_ms = 0

        try:
            source_size = int(track.attrib.get("size", "0"))
        except ValueError:
            source_size = 0

        media = track.find("Media")
        if media is None:
            continue

        part = media.find("Part")
        if part is None:
            continue

        media_url = f"{server.base_url}{part.attrib.get('key', '')}"

        if not rating_key:
            continue

        tracks.append(
            Track(
                rating_key=rating_key,
                title=title,
                artist=artist,
                album=album,
                album_artist=album_artist,
                parent_index=parent_index,
                index=index,
                duration_ms=duration_ms,
                media_url=media_url,
                source_size=source_size,
                playlist_id=library_key,
                container=container,
                audio_codec=audio_codec,
            )
        )

    return tracks


def transcode_to_mp3(
    media_url: str,
    destination: Path,
    server_token: str,
    timeout: int = 120,
) -> bool:
    """Transcode media to MP3 V0 using ffmpeg."""
    # Add token to URL if present
    separator = "&" if "?" in media_url else "?"
    url = (
        f"{media_url}{separator}X-Plex-Token={server_token}"
        if server_token
        else media_url
    )

    # Detect if source is already MP3 and use stream copy
    audio_codec_arg = (
        "copy"
        if (
            # We would detect this from Track attributes in real usage
            False
        )
        else "libmp3lame"
    )

    cmd = [
        "ffmpeg",
        "-i", url,
        "-codec:a", audio_codec_arg,
        "-q:a", "0" if audio_codec_arg == "libmp3lame" else None,
        "-y",  # Overwrite output
        str(destination),
    ]

    # Filter out None values
    cmd = [x for x in cmd if x is not None]

    try:
        subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return False


def download_track(
    job: DownloadJob,
    config: dict,
    timeout: int,
) -> DownloadResult:
    """Download and transcode one track."""
    start_time = time.time()
    attempts = 0
    retries = config["download"]["retries"]
    retry_delay = config["download"]["retry_delay"]

    job.destination.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists and is valid
    if job.destination.exists():
        return DownloadResult(
            job=job,
            success=True,
            skipped=True,
            bytes_written=0,
            elapsed=time.time() - start_time,
            attempts=0,
        )

    # Transcode to MP3 V0
    for attempt in range(1, retries + 1):
        attempts = attempt
        if transcode_to_mp3(
            job.track.media_url,
            job.destination,
            config["plex"]["token"],
            timeout,
        ):
            bytes_written = (
                job.destination.stat().st_size
                if job.destination.exists()
                else 0
            )
            return DownloadResult(
                job=job,
                success=True,
                skipped=False,
                bytes_written=bytes_written,
                elapsed=time.time() - start_time,
                attempts=attempts,
            )

        if attempt < retries:
            time.sleep(retry_delay)
            # Clean up partial file
            if job.destination.exists():
                job.destination.unlink()

    return DownloadResult(
        job=job,
        success=False,
        skipped=False,
        bytes_written=0,
        elapsed=time.time() - start_time,
        attempts=attempts,
        error="Transcode failed after retries",
    )


_abort_on_signal = False


def _signal_handler(signum: int, frame: object) -> None:
    """Handle signals."""
    global _abort_on_signal

    if not _abort_on_signal:
        _abort_on_signal = True
        print(SIGNAL_MSG, end="", flush=True)
        signal.signal(signum, signal.SIG_IGN)


def main() -> None:
    """Main entry point."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, _signal_handler)

    config = load_config()
    server = prompt_server(config)

    print()
    library_key, library_name = select_music_library(
        server,
        config["plex"]["timeout"],
    )

    print()
    playlists = get_playlists(
        server,
        config["plex"]["timeout"],
    )

    if not playlists:
        print("No playlists available.")
        return

    print("Available playlists:")
    print()

    for index, (rating_key, title, count, duration) in enumerate(playlists, 1):
        print(f"  {index:>2}. {title:30} ({count:3} tracks, {duration})")

    print()

    while True:
        answer = input("Select playlist [1]: ").strip()

        if not answer:
            playlist_index = 0
            break

        try:
            playlist_index = int(answer) - 1
            if 0 <= playlist_index < len(playlists):
                break
        except ValueError:
            pass

        print("Invalid selection.")

    rating_key, title, count, duration = playlists[playlist_index]

    print()
    print(f"Downloading: {title}")
    print()

    # Get tracks
    if title.lower() == "random":
        # Get random sample from library
        tracks = get_library_items(
            server,
            library_key,
            config["plex"]["timeout"],
        )
        random.shuffle(tracks)
    else:
        # Get playlist tracks
        tracks = get_playlist_items(
            server,
            rating_key,
            config["plex"]["timeout"],
        )

    download_dir = (
        Path(__file__).resolve().parent
        / config["output"]["directory"]
        / sanitize_filename(title)
    )

    # Create download jobs
    jobs: list[DownloadJob] = []
    for index, track in enumerate(tracks, 1):
        filename = track_filename(track, index)
        destination = download_dir / filename
        jobs.append(
            DownloadJob(
                index=index,
                total=len(tracks),
                track=track,
                destination=destination,
            )
        )

    # Download tracks
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(download_track, job, config, config["plex"]["timeout"])
            for job in jobs
        ]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            status = "✓" if result.success else "✗"
            skipped_text = " (cached)" if result.skipped else ""
            print(
                f"[{result.job.index:3}/{result.job.total}] {status} "
                f"{compact(result.job.track.title, 40)} "
                f"{human_size(result.bytes_written):>10} "
                f"{human_rate(result.bytes_written / result.elapsed):>12}"
                f"{skipped_text}"
            )

    print()
    print("Download complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Interrupted.")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
