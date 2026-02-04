import datetime
import logging
import os
import pathlib
import re
import shutil
import stat
import subprocess
import time
import urllib.parse

from etils import epath
import filelock
import tensorflow as tf

_OPENPI_DATA_HOME = "OPENPI_DATA_HOME"
_CACHE_COMPLETE_MARKER = "COMMIT_SUCCESS"
_LEGACY_COMPLETE_MARKER = "commit_success.txt"

logger = logging.getLogger(__name__)


def get_cache_dir() -> pathlib.Path | epath.Path:
    """Return the cache directory, creating it if necessary.

    Environment variable `OPENPI_DATA_HOME` must point to either a local POSIX path
    or a `gs://` URI.
    """
    cache_dir_str = os.getenv(_OPENPI_DATA_HOME)
    if cache_dir_str is None:
        raise ValueError(f"Environment variable {_OPENPI_DATA_HOME} must be set.")

    if _is_gcs(cache_dir_str):
        cache_dir = epath.Path(cache_dir_str)
        tf.io.gfile.makedirs(str(cache_dir))
        return cache_dir

    cache_dir = pathlib.Path(cache_dir_str)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def maybe_download(
    url: str,
    *,
    force_download: bool = False,
    **kwargs,
) -> pathlib.Path | epath.Path:
    """Return a local/GCS cache path for ``url``, downloading on cache miss."""
    parsed = urllib.parse.urlparse(url)

    if parsed.scheme == "":
        return _resolve_local_input(url)

    if parsed.scheme != "gs":
        raise NotImplementedError(f"Downloading from non-GCS URLs is not supported. Got: {url}")

    same_bucket_source = _same_bucket_source(url)
    if same_bucket_source is not None:
        return same_bucket_source

    cache_dir = get_cache_dir()
    remote_cache = _is_gcs(cache_dir)
    cache_path = _build_cache_path(cache_dir, parsed)
    scratch_path = f"{cache_path}.partial"
    lock_path = f"{cache_path}.lock"

    cache_hit, invalidate_cache = _check_cache_status(
        cache_dir=cache_dir,
        cache_path=cache_path,
        remote_cache=remote_cache,
        force_download=force_download,
    )
    if cache_hit:
        logger.info("Cache hit: %s", cache_path)
        return epath.Path(cache_path) if remote_cache else pathlib.Path(cache_path)

    _remove_path_if_exists(scratch_path, remote_cache=remote_cache)

    lock = filelock.FileLock(lock_path) if not remote_cache else None
    try:
        if lock:
            lock.acquire()

        if invalidate_cache:
            _remove_path_if_exists(cache_path, remote_cache=remote_cache)

        logger.info("Downloading %s to %s", url, scratch_path)
        _download_gcs(url, scratch_path, **kwargs)

        _write_completion_marker_if_directory(scratch_path, remote_cache=remote_cache)
        _promote_scratch_to_cache(scratch_path, cache_path, remote_cache=remote_cache)

        if not remote_cache:
            _ensure_permissions(pathlib.Path(cache_path))

    except PermissionError as e:
        msg = f"Permission error while downloading {url}. Try removing the cache entry: rm -rf {cache_path}*"
        raise PermissionError(msg) from e
    finally:
        if lock and lock.is_locked:
            lock.release()

    return epath.Path(cache_path) if remote_cache else pathlib.Path(cache_path)


def ensure_commit_success(dir_path: str) -> None:
    """Ensure completion markers exist for a local or ``gs://`` directory.

    Best-effort: failures are intentionally ignored.
    """
    try:
        if _is_gcs(dir_path):
            if not tf.io.gfile.isdir(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")
            _write_remote_file_if_missing(_join(dir_path, _CACHE_COMPLETE_MARKER), "ok")
            _write_remote_file_if_missing(_join(dir_path, _LEGACY_COMPLETE_MARKER), "ok")
            return

        marker_dir = pathlib.Path(dir_path)
        if not marker_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        _write_local_file_if_missing(marker_dir / _CACHE_COMPLETE_MARKER, "ok")
        _write_local_file_if_missing(marker_dir / _LEGACY_COMPLETE_MARKER, "ok")
    except Exception:
        pass


def _resolve_local_input(url: str) -> pathlib.Path:
    path = pathlib.Path(url)
    if not path.exists():
        raise FileNotFoundError(f"File not found at {url}")
    return path.resolve()


def _is_gcs(path: str | pathlib.Path | epath.Path) -> bool:
    return str(path).startswith("gs://")


def _join(*parts: str | pathlib.Path | epath.Path) -> str:
    """Join paths for either local filesystem or GCS."""
    normalized = [str(p) for p in parts]
    if _is_gcs(normalized[0]):
        return tf.io.gfile.join(*normalized)
    return str(pathlib.Path(normalized[0], *normalized[1:]))


def _same_bucket_source(url: str) -> epath.Path | None:
    """Return source path when cache/source share bucket; validate existence first."""
    cache_dir_probe = os.getenv(_OPENPI_DATA_HOME)
    if not (cache_dir_probe and _is_gcs(cache_dir_probe)):
        return None

    cache_bucket = urllib.parse.urlparse(cache_dir_probe).netloc
    source_bucket = urllib.parse.urlparse(url).netloc
    if cache_bucket != source_bucket:
        return None

    try:
        if tf.io.gfile.isdir(url) or tf.io.gfile.exists(url):
            return epath.Path(url)
    except tf.errors.NotFoundError as e:
        raise FileNotFoundError(f"File not found at {url}") from e
    raise FileNotFoundError(f"File not found at {url}")


def _build_cache_path(cache_dir: pathlib.Path | epath.Path, parsed_url: urllib.parse.ParseResult) -> str:
    return _join(cache_dir, parsed_url.netloc, parsed_url.path.lstrip("/"))


def _check_cache_status(
    *,
    cache_dir: pathlib.Path | epath.Path,
    cache_path: str,
    remote_cache: bool,
    force_download: bool,
) -> tuple[bool, bool]:
    """Return ``(cache_hit, invalidate_cache)``."""
    if not _path_exists(cache_path, remote_cache=remote_cache):
        return False, False

    if force_download:
        return False, True

    if remote_cache:
        if _is_complete_remote_cache_entry(cache_path):
            return True, False
        return False, True

    local_cache_dir = pathlib.Path(str(cache_dir))
    local_cache_path = pathlib.Path(cache_path)
    if _should_invalidate_cache(local_cache_dir, local_cache_path):
        return False, True
    return True, False


def _is_complete_remote_cache_entry(path: str) -> bool:
    if not tf.io.gfile.exists(path):
        return False
    if not tf.io.gfile.isdir(path):
        return True

    return any(
        tf.io.gfile.exists(_join(path, marker))
        for marker in ("_METADATA", _CACHE_COMPLETE_MARKER, _LEGACY_COMPLETE_MARKER)
    )


def _path_exists(path: str, *, remote_cache: bool) -> bool:
    if remote_cache:
        return tf.io.gfile.exists(path)
    return pathlib.Path(path).exists()


def _remove_path_if_exists(path: str, *, remote_cache: bool) -> None:
    if not _path_exists(path, remote_cache=remote_cache):
        return

    logger.info("Removing path: %s", path)
    if remote_cache:
        try:
            if tf.io.gfile.isdir(path):
                tf.io.gfile.rmtree(path)
            else:
                tf.io.gfile.remove(path)
        except tf.errors.NotFoundError:
            return
        return

    local_path = pathlib.Path(path)
    if local_path.is_dir():
        shutil.rmtree(local_path)
    else:
        local_path.unlink()


def _write_completion_marker_if_directory(path: str, *, remote_cache: bool) -> None:
    """Mark downloaded directories as complete (best-effort)."""
    try:
        if remote_cache:
            if tf.io.gfile.isdir(path):
                _write_remote_file_if_missing(_join(path, _CACHE_COMPLETE_MARKER), "ok")
                _write_remote_file_if_missing(_join(path, _LEGACY_COMPLETE_MARKER), "ok")
        else:
            local_path = pathlib.Path(path)
            if local_path.is_dir():
                _write_local_file_if_missing(local_path / _CACHE_COMPLETE_MARKER, "ok")
                _write_local_file_if_missing(local_path / _LEGACY_COMPLETE_MARKER, "ok")
    except Exception:
        pass


def _write_remote_file_if_missing(path: str, content: str) -> None:
    if tf.io.gfile.exists(path):
        return
    with tf.io.gfile.GFile(path, "w") as f:
        f.write(content)


def _write_local_file_if_missing(path: pathlib.Path, content: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _promote_scratch_to_cache(scratch_path: str, cache_path: str, *, remote_cache: bool) -> None:
    if remote_cache:
        try:
            tf.io.gfile.rename(scratch_path, cache_path, overwrite=True)
        except tf.errors.NotFoundError as e:
            # GCS cache writes can race across workers. If another worker already
            # finalized this cache path, treat this as success.
            if tf.io.gfile.exists(cache_path):
                logger.info(
                    "Scratch path disappeared during promote; using existing cache entry: %s",
                    cache_path,
                )
                return
            raise e
        return
    shutil.move(scratch_path, cache_path)


def _download_gcs(url: str, destination_path: pathlib.Path | str, **kwargs) -> None:
    """Download a GCS file/dir using ``gsutil``.

    ``kwargs`` is accepted for API compatibility with callers that pass through
    fsspec-style options (for example ``gs={"token": "anon"}``).
    """
    del kwargs
    destination = str(destination_path)

    if _is_gcs_directory(url):
        _download_gcs_directory(url, destination)
    else:
        _download_gcs_file(url, destination)


def _is_gcs_directory(url: str) -> bool:
    """Return True when ``url`` points to a GCS prefix instead of a single object."""
    if tf.io.gfile.isdir(url):
        return True

    stat_result = _run_gsutil(["stat", url])
    if stat_result.returncode == 0:
        return False

    list_result = _run_gsutil(["ls", f"{url.rstrip('/')}/"])
    return list_result.returncode == 0 and bool(list_result.stdout.strip())


def _download_gcs_file(url: str, destination: str) -> None:
    parent_dir = os.path.dirname(destination)
    if parent_dir:
        if _is_gcs(destination):
            tf.io.gfile.makedirs(parent_dir)
        else:
            pathlib.Path(parent_dir).mkdir(parents=True, exist_ok=True)

    result = _run_gsutil(["cp", url, destination])
    _raise_on_gsutil_error(result, "cp")


def _download_gcs_directory(url: str, destination: str) -> None:
    if _is_gcs(destination):
        tf.io.gfile.makedirs(destination)
    else:
        pathlib.Path(destination).mkdir(parents=True, exist_ok=True)

    result = _run_gsutil(["-m", "rsync", "-r", url, destination])
    _raise_on_gsutil_error(result, "rsync")


def _run_gsutil(args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = ["gsutil", *args]
    logger.info("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _raise_on_gsutil_error(result: subprocess.CompletedProcess[str], operation: str) -> None:
    if result.returncode == 0:
        return
    raise RuntimeError(
        f"gsutil {operation} failed with return code {result.returncode}.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


def _set_permission(path: pathlib.Path, target_permission: int) -> None:
    """Apply ``target_permission`` if it is not already fully present."""
    if path.stat().st_mode & target_permission == target_permission:
        logger.debug("Skipping %s because it already has correct permissions", path)
        return
    path.chmod(target_permission)


def _set_folder_permission(folder_path: pathlib.Path) -> None:
    """Set folder permission to read, write, and searchable for all users."""
    _set_permission(folder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)


def _ensure_permissions(path: pathlib.Path) -> None:
    """Ensure cache permissions are shared across training/containerized runtimes."""

    cache_dir = get_cache_dir()
    if not isinstance(cache_dir, pathlib.Path):
        return

    try:
        relative_path = path.relative_to(cache_dir)
    except ValueError:
        return
    moving_path = cache_dir
    for part in relative_path.parts:
        moving_path = moving_path / part
        _set_folder_permission(moving_path)

    file_rw = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH
    for root, dirs, files in os.walk(str(path)):
        root_path = pathlib.Path(root)
        for file_name in files:
            file_path = root_path / file_name
            is_executable = bool(file_path.stat().st_mode & stat.S_IXUSR)
            target_mode = file_rw | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH if is_executable else file_rw
            _set_permission(file_path, target_mode)

        for dir_name in dirs:
            _set_folder_permission(root_path / dir_name)


def _get_mtime(year: int, month: int, day: int) -> float:
    """Get mtime for midnight UTC on a given date."""
    date = datetime.datetime(year, month, day, tzinfo=datetime.UTC)
    return time.mktime(date.timetuple())


_INVALIDATE_CACHE_DIRS: dict[re.Pattern[str], float] = {
    re.compile("openpi-assets/checkpoints/pi0_aloha_pen_uncap"): _get_mtime(2025, 2, 17),
    re.compile("openpi-assets/checkpoints/pi0_libero"): _get_mtime(2025, 2, 6),
    re.compile("openpi-assets/checkpoints/"): _get_mtime(2025, 2, 3),
}


def _should_invalidate_cache(cache_dir: pathlib.Path, local_path: pathlib.Path) -> bool:
    """Return True when a local cache entry is older than its invalidation date."""
    assert local_path.exists(), f"File not found at {local_path}"

    relative_path = str(local_path.relative_to(cache_dir))
    for pattern, expire_time in _INVALIDATE_CACHE_DIRS.items():
        if pattern.match(relative_path):
            return local_path.stat().st_mtime <= expire_time

    return False
