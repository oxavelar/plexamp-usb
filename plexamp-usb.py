#!/usr/bin/env python3
"""Export Plex music to a car-friendly USB filesystem.

The program discovers a reachable Plex Media Server, authenticates only when
local unauthenticated access is unavailable, lets the user select Plex music
playlists, transcodes tracks to MP3 V0 VBR, and writes them below Downloads
beside this script.

Downloads are deterministic and resumable. Valid existing files are retained;
invalid or incomplete files are replaced. Directories can be capped at a
user-selected number of audio files.

A playlist named "Random" enables fill mode. Tracks are sampled from the full
Plex music library until the destination filesystem reaches its safety reserve.
Existing exported tracks are skipped so repeated runs continue adding music.

Configuration is read from settings.json. No Plex API client library is used.

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
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
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
    """Fit text to a fixed console width."""
    text = str(text)

    if len(text) <= width:
        return text

    return text[: max(0, width - 1)] + "…" if width > 1 else text[:width]


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
    """Format a Plex duration for console display."""
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
    """Return a deterministic filename component safe for common USB filesystems."""
    value = unicodedata.normalize("NFC", str(value or ""))
    value = value.replace("\x00", "")
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

    if len(value.encode("utf-8")) <= max_bytes:
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


def track_filename(track: Track, number: int) -> str:
    """Build a stable car-friendly MP3 filename."""
    artist = sanitize_filename(track.artist, "Unknown Artist")
    album = sanitize_filename(track.album, "Unknown Album")
    title = sanitize_filename(track.title, "Unknown Track")

    try:
        track_number = int(track.index)
    except (TypeError, ValueError):
        track_number = 0

    prefix = f"{track_number:02d}" if track_number else f"{number:02d}"
    prefix_part = f"{prefix} - {artist} - {album} - "
    extension = ".mp3"

    available = 255 - len((prefix_part + extension).encode("utf-8"))

    if available <= 0:
        album = sanitize_filename(album, "Unknown Album", 80)
        prefix_part = f"{prefix} - {artist} - {album} - "
        available = 255 - len((prefix_part + extension).encode("utf-8"))

    title = sanitize_filename(
        title,
        "Unknown Track",
        max(1, available),
    )

    return f"{prefix_part}{title}{extension}"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge user configuration over defaults."""
    result = dict(base)

    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def config_path() -> Path:
    return Path(__file__).resolve().parent / CONFIG_JSON


def save_default_config(path: Path) -> None:
    """Create the initial configuration file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            DEFAULT_CONFIG,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")


def load_config() -> dict:
    """Load settings.json, creating it from defaults when absent."""
    path = config_path()

    if not path.exists():
        save_default_config(path)
        print(f"  Created {path}")
        return deep_merge({}, DEFAULT_CONFIG)

    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise RuntimeError(f"Invalid configuration: {path}")

    return deep_merge(DEFAULT_CONFIG, loaded)


def build_request(
    url: str,
    token: str = "",
    accept: str = "*/*",
) -> urllib.request.Request:
    """Build a Plex HTTP request."""
    headers = {
        "User-Agent": f"{APP_NAME}/1.0",
        "Accept": accept,
    }

    if token:
        headers["X-Plex-Token"] = token

    return urllib.request.Request(url, headers=headers)


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
    """Request and parse a Plex XML endpoint."""
    query = urllib.parse.urlencode(params or {})
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
    """Return the Plex identity when the server is reachable."""
    try:
        return plex_xml(server, "/identity", timeout=timeout)
    except Exception:
        return None


def discover_gdm_servers(timeout: float = 3.0) -> list[PlexServer]:
    """Discover local Plex servers using Plex GDM."""
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

            if not data:
                continue

            headers = {}

            for line in data.decode(
                "utf-8",
                errors="replace",
            ).splitlines()[1:]:
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
            name = identity.attrib.get(
                "friendlyName",
                configured_host,
            )

            print(f"  Server: {name}")
            print(f"  Address: {configured.base_url}")
            print("  Access:  OK")

            return PlexServer(
                name=name,
                host=configured.host,
                port=configured.port,
                protocol=configured.protocol,
                token=configured.token,
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

        address = input("  Plex server address: ").strip()

        if not address:
            raise RuntimeError("No Plex server address supplied.")

        parsed = urllib.parse.urlparse(address)

        if parsed.scheme:
            protocol = parsed.scheme
            host = parsed.hostname or address
            port = parsed.port or configured_port
        else:
            protocol = "http"

            if ":" in address:
                host, possible_port = address.rsplit(":", 1)

                try:
                    port = int(possible_port)
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

        for index, candidate in enumerate(servers, 1):
            print(
                f"    {index}. "
                f"{candidate.name} "
                f"({candidate.base_url})"
            )

        while True:
            answer = input("\n  Select Plex server [1]: ").strip()

            if not answer:
                server = servers[0]
                break

            try:
                index = int(answer)
            except ValueError:
                index = 0

            if 1 <= index <= len(servers):
                server = servers[index - 1]
                break

            print("  Invalid selection.")

    print(f"  Found: {server.name} ({server.base_url})")

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

    token = configured_token or input("  Plex token: ").strip()

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

    libraries = [
        (
            directory.attrib.get("key", ""),
            directory.attrib.get("title", "Music"),
        )
        for directory in root.findall("Directory")
        if directory.attrib.get("type") == "artist"
        and directory.attrib.get("key")
    ]

    if not libraries:
        raise RuntimeError("No Plex music library was found.")

    if len(libraries) == 1:
        return libraries[0]

    print()
    print("Music libraries:")
    print()

    for index, (_, title) in enumerate(libraries, 1):
        print(f"  {index:>2}. {title}")

    while True:
        answer = input("\nSelect music library [1]: ").strip()

        if not answer:
            return libraries[0]

        try:
            index = int(answer)
        except ValueError:
            index = 0

        if 1 <= index <= len(libraries):
            return libraries[index - 1]

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

        if not rating_key:
            continue

        try:
            count = int(playlist.attrib.get("leafCount", 0))
        except ValueError:
            count = 0

        playlists.append(
            (
                rating_key,
                playlist.attrib.get("title", "Unnamed"),
                count,
                human_duration(playlist.attrib.get("duration")),
            )
        )

    return playlists


def choose_playlists(
    playlists: list[tuple[str, str, int, str]],
) -> list[tuple[str, str]]:
    """Prompt for one or more playlist downloads."""
    if not playlists:
        raise RuntimeError("No Plex music playlists were found.")

    print()
    print("Music playlists:")
    print()

    for index, (_, title, count, duration) in enumerate(
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
        answer = input("\nSelect downloads: ").strip()

        if answer.lower() == "a":
            return [(key, title) for key, title, _, _ in playlists]

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
                (playlists[index - 1][0], playlists[index - 1][1])
                for index in indices
            ]

            if random_selected:
                selected.append(("Random", "Random"))

            if selected:
                return selected
        except ValueError:
            pass

        print("Invalid selection.")


def parse_track(
    item: ET.Element,
    server: PlexServer,
    playlist_id: str,
) -> Track | None:
    """Convert one Plex track element into a Track."""
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
        album=item.attrib.get("parentTitle", "Unknown Album"),
        album_artist=item.attrib.get("parentTitle", ""),
        parent_index=item.attrib.get("parentIndex", "1"),
        index=item.attrib.get("index", "0"),
        duration_ms=duration_ms,
        media_url=urllib.parse.urljoin(server.base_url, key),
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
    """Fetch playable audio tracks from a Plex playlist."""
    root = plex_xml(
        server,
        f"/playlists/{playlist_id}/items",
        timeout=timeout,
    )

    return [
        track
        for item in root
        if (track := parse_track(item, server, playlist_id)) is not None
    ]


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

    return [
        track
        for item in root.findall("Track")
        if (track := parse_track(item, server, "Random")) is not None
    ]


def unique_tracks(tracks: Iterable[Track]) -> list[Track]:
    """Remove duplicate tracks while preserving order."""
    result = []
    seen = set()

    for track in tracks:
        identity = track.rating_key or (
            f"{track.media_url}|"
            f"{track.title}|"
            f"{track.artist}|"
            f"{track.album}"
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
        except ValueError:
            value = 0

        if value == -1 or value > 0:
            return value

        print("  Enter a positive number or -1.")


def playlist_directory(name: str) -> str:
    return sanitize_filename(name, "Music")


def build_output_path(
    root: Path,
    playlist_name: str,
    position: int,
    track: Track,
    directory_limit: int,
) -> Path:
    """Build the deterministic destination for a track."""
    playlist_root = root / playlist_directory(playlist_name)

    if directory_limit == -1:
        directory = playlist_root
        directory_position = position
    else:
        directory_number, directory_position = divmod(
            position - 1,
            directory_limit,
        )
        directory = playlist_root / f"{directory_number + 1:03d}"
        directory_position += 1

    return directory / track_filename(track, directory_position)


def file_is_valid(path: Path) -> bool:
    """Verify an existing MP3 by decoding its first audio stream."""
    try:
        if not path.is_file() or path.stat().st_size < 1024:
            return False
    except OSError:
        return False

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
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False

    return result.returncode == 0


def check_ffmpeg() -> None:
    """Verify that ffmpeg is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(
            "ffmpeg is required but was not found in PATH."
        ) from exc

    if result.returncode != 0:
        raise RuntimeError("ffmpeg could not be executed.")


def determine_workers() -> int:
    """Choose bounded parallelism for Plex, ffmpeg, and USB I/O."""
    return max(2, min(8, os.cpu_count() or 2))


def free_space(path: Path) -> int:
    """Return available bytes on the destination filesystem."""
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0


def random_fill_space_available(output_root: Path) -> bool:
    return free_space(output_root) > RANDOM_FILL_RESERVE


def track_identity(track: Track) -> str:
    """Return a normalized artist/album/title identity."""
    return "|".join(
        sanitize_filename(value, "").casefold()
        for value in (
            track.artist,
            track.album,
            track.title,
        )
    )


def exported_track_keys(output_root: Path) -> set[str]:
    """Read exported filenames to avoid obvious Random-mode duplicates."""
    keys = set()

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


def select_random_tracks(
    tracks: list[Track],
    output_root: Path,
) -> list[Track]:
    """Shuffle library tracks and exclude already exported identities."""
    existing = exported_track_keys(output_root)

    candidates = [
        track
        for track in tracks
        if track_identity(track) not in existing
    ]

    random.SystemRandom().shuffle(candidates)
    return candidates


def ffmpeg_command(
    url: str,
    output: Path,
    token: str,
    copy: bool = False,
) -> list[str]:
    """Build the FFmpeg command used for one atomic transfer."""
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
            "-c:a",
            "copy" if copy else "libmp3lame",
        ]
    )

    if not copy:
        command.extend(["-q:a", "0"])

    command.extend(
        [
            "-f",
            "mp3",
            str(output),
        ]
    )

    return command


def run_ffmpeg(
    url: str,
    destination: Path,
    token: str,
    timeout: int,
    copy: bool = False,
) -> tuple[bool, int, str]:
    """Write FFmpeg output to a temporary file and atomically replace the destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    part_path = destination.with_name(
        destination.name + ".part"
    )

    try:
        part_path.unlink(missing_ok=True)
    except OSError:
        pass

    try:
        result = subprocess.run(
            ffmpeg_command(
                url,
                part_path,
                token,
                copy,
            ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(60, timeout) * 30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        part_path.unlink(missing_ok=True)
        return False, 0, "ffmpeg timed out"
    except OSError as exc:
        return False, 0, str(exc)

    if result.returncode != 0:
        error = result.stderr.strip() or (
            f"ffmpeg exited with code {result.returncode}"
        )
        part_path.unlink(missing_ok=True)
        return False, 0, error[-1600:]

    try:
        size = part_path.stat().st_size
    except OSError:
        size = 0

    if size < 1024:
        part_path.unlink(missing_ok=True)
        return False, 0, "ffmpeg produced an empty or incomplete MP3"

    if not file_is_valid(part_path):
        part_path.unlink(missing_ok=True)
        return False, 0, "MP3 validation failed after transfer"

    try:
        os.replace(part_path, destination)
    except OSError as exc:
        part_path.unlink(missing_ok=True)
        return False, 0, str(exc)

    return True, size, ""


def download_track(
    job: DownloadJob,
    server: PlexServer,
    retries: int,
    retry_delay: float,
    timeout: int,
) -> DownloadResult:
    """Download one track with resumable validation and exponential retry."""
    destination = job.destination

    if file_is_valid(destination):
        try:
            size = destination.stat().st_size
        except OSError:
            size = 0

        return DownloadResult(
            job=job,
            success=True,
            skipped=True,
            bytes_written=size,
            elapsed=0,
            attempts=0,
        )

    started = time.monotonic()
    source_is_mp3 = (
        job.track.container.lower() == "mp3"
        or job.track.audio_codec.lower() == "mp3"
        or job.track.media_url.lower().endswith(".mp3")
    )

    last_error = "unknown error"

    for attempt in range(1, retries + 1):
        success, size, error = run_ffmpeg(
            url=job.track.media_url,
            destination=destination,
            token=server.token,
            timeout=timeout,
            copy=source_is_mp3,
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

        if attempt == retries:
            break

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

    print(
        f"  [{result.job.index:>3}/"
        f"{result.job.total:<3}] "
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
            f"       ! "
            f"{compact(' '.join(result.error.split()), 140)}"
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
                output_root,
                playlist_name,
                index,
                track,
                directory_limit,
            ),
        )
        for index, track in enumerate(tracks, 1)
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
) -> tuple[int, int, int]:
    """Download all tracks from one playlist in parallel."""
    if not tracks:
        return 0, 0, 0

    jobs = create_jobs(
        tracks,
        playlist_name,
        output_root,
        directory_limit,
    )
    workers = determine_workers()

    print()
    print(f"  {playlist_name}")
    print(f"  {'─' * min(72, len(playlist_name) + 2)}")
    print(f"  Tracks:     {len(jobs):,}")
    print(f"  Parallel:   {workers}")
    print("  Encoding:   MP3 V0 VBR")
    print()

    downloaded = failed = written = 0

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="plex",
    ) as executor:
        futures = [
            executor.submit(
                download_track,
                job,
                server,
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

            print_result(result, output_root)

    return downloaded, failed, written


def download_random_fill(
    tracks: list[Track],
    output_root: Path,
    directory_limit: int,
    server: PlexServer,
    retries: int,
    retry_delay: float,
    timeout: int,
) -> tuple[int, int, int]:
    """Fill available USB capacity with randomly selected new tracks."""
    candidates = select_random_tracks(tracks, output_root)

    if not candidates:
        print()
        print("  Random: no new tracks remain.")
        return 0, 0, 0

    workers = determine_workers()
    random_root = output_root / playlist_directory("Random")

    existing_count = (
        sum(1 for path in random_root.rglob("*.mp3"))
        if random_root.exists()
        else 0
    )

    print()
    print("  Random")
    print("  ──────")
    print(f"  Library tracks:   {len(tracks):,}")
    print(f"  New candidates:   {len(candidates):,}")
    print(f"  Parallel:         {workers}")
    print("  Encoding:         MP3 V0 VBR")
    print("  Fill mode:        until drive capacity")
    print(f"  Safety reserve:   {human_size(RANDOM_FILL_RESERVE)}")
    print()

    downloaded = failed = written = 0
    position = existing_count + 1
    cursor = 0

    while (
        cursor < len(candidates)
        and random_fill_space_available(output_root)
    ):
        batch = candidates[cursor:cursor + workers]
        cursor += len(batch)

        jobs = [
            DownloadJob(
                index=position + offset,
                total=0,
                track=track,
                destination=build_output_path(
                    output_root,
                    "Random",
                    position + offset,
                    track,
                    directory_limit,
                ),
            )
            for offset, track in enumerate(batch)
        ]

        position += len(batch)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="random",
        ) as executor:
            futures = [
                executor.submit(
                    download_track,
                    job,
                    server,
                    retries,
                    retry_delay,
                    timeout,
                )
                for job in jobs
            ]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()

                print_random_result(
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


def print_random_result(
    result: DownloadResult,
    output_root: Path,
) -> None:
    """Print one Random Fill result."""
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

    print(
        f"  [{result.job.index:>4}] "
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
            f"       ! "
            f"{compact(' '.join(result.error.split()), 140)}"
        )


def main() -> int:
    """Run the interactive Plex music export workflow."""
    print()
    print("=" * 78)
    print("                              plexamp-usb")
    print("=" * 78)
    print()
    print(
        "  Export Plex music downloads to a "
        "car-friendly USB filesystem."
    )
    print()

    try:
        check_ffmpeg()
        config = load_config()
        server = prompt_server(config)

        timeout = int(config["plex"].get("timeout", 30))

        library_key, section_name = select_music_library(
            server,
            timeout,
        )

        print()
        print(f"Music library: {section_name}")

        selected = choose_playlists(
            get_playlists(server, timeout)
        )

        directory_limit = choose_directory_limit()

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

        workers = determine_workers()
        has_random = any(
            title.casefold().strip() == "random"
            for _, title in selected
        )

        print()
        print("Download configuration")
        print()
        print(f"  Server:       {server.name}")
        print(f"  Address:      {server.base_url}")
        print("  Output:       MP3 V0 VBR")
        print(f"  Parallel:     {workers} workers")
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

        output_root.mkdir(parents=True, exist_ok=True)

        selected_tracks: list[tuple[str, list[Track]]] = []
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

            print(f"{len(random_tracks):>6,} tracks")

        print()

        grand_downloaded = 0
        grand_failed = 0
        grand_written = 0

        # Explicit playlists run first so Random Fill only consumes capacity
        # left after the user's requested downloads.
        for playlist_name, tracks in selected_tracks:
            if not tracks:
                print(
                    f"  Skipping empty download: "
                    f"{playlist_name}"
                )
                continue

            downloaded, failed, written = download_playlist(
                tracks,
                playlist_name,
                output_root,
                directory_limit,
                server,
                retries,
                retry_delay,
                timeout,
            )

            grand_downloaded += downloaded
            grand_failed += failed
            grand_written += written

        if random_selected:
            downloaded, failed, written = download_random_fill(
                random_tracks,
                output_root,
                directory_limit,
                server,
                retries,
                retry_delay,
                timeout,
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
        print("Aborting...", end="", flush=True)
        print("\r\033[KInterrupted.")
        return 130

    except Exception as exc:
        print()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
