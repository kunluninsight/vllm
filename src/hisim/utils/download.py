"""Fetch remote files into a local cache shared by simulator processes."""

import contextlib
import fcntl
import hashlib
import http.client
import os
import shutil
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen

from hisim.utils.logger import get_logger

logger = get_logger()

_CACHE_DIR = "/tmp/hisim/downloads"
_TIMEOUT_S = 60


def resolve_local_path(path: str) -> str:
    """Return ``path`` unchanged, or a cached local copy if it is an HTTP(S) URL."""
    if not isinstance(path, str) or not path.lower().startswith(
        ("http://", "https://")
    ):
        return path

    parts = urlsplit(path)
    # Query strings and userinfo may hold credentials, so never log them.
    display_url = f"{parts.scheme}://{parts.hostname}{parts.path}"
    digest = hashlib.sha256(path.encode()).hexdigest()[:16]
    local_path = os.path.join(_CACHE_DIR, f"{digest}-{os.path.basename(parts.path)}")

    os.makedirs(_CACHE_DIR, exist_ok=True)
    # Workers start together; the lock makes them share a single download.
    with open(f"{local_path}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.exists(local_path):
            logger.info("Using cached %s for %s", local_path, display_url)
            return local_path
        logger.info("Downloading %s to %s", display_url, local_path)
        try:
            _download(path, local_path)
        except (OSError, http.client.HTTPException) as e:
            raise RuntimeError(f"Failed to download {display_url}: {e}") from e
    return local_path


def _download(url: str, local_path: str) -> None:
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(local_path))
    try:
        with os.fdopen(fd, "wb") as f, urlopen(url, timeout=_TIMEOUT_S) as resp:
            shutil.copyfileobj(resp, f)
            # A dropped connection can end the stream early without an error.
            expected = resp.headers.get("Content-Length")
            if expected is not None and f.tell() != int(expected):
                raise OSError(f"got {f.tell()} of {expected} bytes")
        os.replace(tmp_path, local_path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)
