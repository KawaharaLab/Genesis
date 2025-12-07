import glob
import gzip
import json
import multiprocessing
import os
import random
import shutil
import time
import urllib.request
import warnings
from typing import Any, Dict, List, Optional, Tuple

from urllib.error import HTTPError, URLError

from tqdm import tqdm

BASE_PATH = os.path.join(os.path.expanduser("~"), ".objaverse")

__version__ = "<REPLACE_WITH_VERSION>"
_VERSIONED_PATH = os.path.join(BASE_PATH, "hf-objaverse-v1")
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OBJECTS_BASE_PATH = os.path.join(_REPO_ROOT, "data", "objects", "hugging_face")


def load_annotations(uids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Load the full metadata of all objects in the dataset.

    Args:
        uids: A list of uids with which to load metadata. If None, it loads
        the metadata for all uids.
    """
    metadata_path = os.path.join(_VERSIONED_PATH, "metadata")
    object_paths = _load_object_paths()
    dir_ids = (
        set([object_paths[uid].split("/")[1] for uid in uids])
        if uids is not None
        else [f"{i // 1000:03d}-{i % 1000:03d}" for i in range(160)]
    )
    if len(dir_ids) > 10:
        dir_ids = tqdm(dir_ids)
    out = {}
    for i_id in dir_ids:
        json_file = f"{i_id}.json.gz"
        local_path = os.path.join(metadata_path, json_file)
        if not os.path.exists(local_path):
            hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/metadata/{i_id}.json.gz"
            # wget the file and put it in local_path
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            _urlretrieve_with_retry(hf_url, local_path)
        with gzip.open(local_path, "rb") as f:
            data = json.load(f)
        if uids is not None:
            data = {uid: data[uid] for uid in uids if uid in data}
        out.update(data)
        if uids is not None and len(out) == len(uids):
            break
    return out


def _urlretrieve_with_retry(
    url: str,
    destination: str,
    *,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> None:
    """Download a URL to the given destination with retry/backoff on transient errors."""

    attempt = 0
    while True:
        try:
            urllib.request.urlretrieve(url, destination)
            return
        except HTTPError as exc:
            should_retry = exc.code in {429, 500, 502, 503, 504}
            if not should_retry or attempt >= max_retries:
                raise

            retry_after_header = exc.headers.get("Retry-After")
            delay = _compute_delay(
                attempt=attempt,
                initial_delay=initial_delay,
                backoff_factor=backoff_factor,
                retry_after_header=retry_after_header if exc.code == 429 else None,
            )
        except URLError:
            if attempt >= max_retries:
                raise
            delay = _compute_delay(
                attempt=attempt,
                initial_delay=initial_delay,
                backoff_factor=backoff_factor,
                retry_after_header=None,
            )

        attempt += 1
        warnings.warn(
            f"Retrying download from {url} in {delay:.1f}s "
            f"(attempt {attempt} of {max_retries})."
        )
        time.sleep(delay + random.uniform(0, 0.25 * delay))


def _compute_delay(
    *,
    attempt: int,
    initial_delay: float,
    backoff_factor: float,
    retry_after_header: Optional[str],
) -> float:
    if retry_after_header:
        try:
            return float(retry_after_header)
        except ValueError:
            pass
    return initial_delay * (backoff_factor**attempt)


def _ensure_local_object(uid: str, object_path: str) -> str:
    """Return the desired local path for an object, migrating legacy layout if needed."""

    local_path = _local_object_path(uid)
    if os.path.exists(local_path):
        return local_path

    legacy_path = os.path.join(_VERSIONED_PATH, object_path)
    if os.path.exists(legacy_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.move(legacy_path, local_path)
    return local_path


def _local_object_path(uid: str) -> str:
    target_dir = os.path.join(OBJECTS_BASE_PATH, uid)
    return os.path.join(target_dir, f"{uid}.glb")


def _load_object_paths() -> Dict[str, str]:
    """Load the object paths from the dataset.

    The object paths specify the location of where the object is located
    in the Hugging Face repo.

    Returns:
        A dictionary mapping the uid to the object path.
    """
    object_paths_file = "object-paths.json.gz"
    local_path = os.path.join(_VERSIONED_PATH, object_paths_file)
    if not os.path.exists(local_path):
        hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_paths_file}"
        # wget the file and put it in local_path
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        _urlretrieve_with_retry(hf_url, local_path)
    with gzip.open(local_path, "rb") as f:
        object_paths = json.load(f)
    return object_paths


def load_uids() -> List[str]:
    """Load the uids from the dataset.

    Returns:
        A list of uids.
    """
    return list(_load_object_paths().keys())


def _download_object(
    uid: str,
    object_path: str,
    total_downloads: float,
    start_file_count: int,
) -> Tuple[str, str]:
    """Download the object for the given uid.

    Args:
        uid: The uid of the object to load.
        object_path: The path to the object in the Hugging Face repo.

    Returns:
        The local path of where the object was downloaded.
    """
    # print(f"downloading {uid}")
    local_path = _local_object_path(uid)
    tmp_local_path = local_path + ".tmp"
    hf_url = (
        f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_path}"
    )
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    _urlretrieve_with_retry(hf_url, tmp_local_path)

    os.rename(tmp_local_path, local_path)

    files = glob.glob(os.path.join(OBJECTS_BASE_PATH, "*", "*.glb"))
    print(
        "Downloaded",
        len(files) - start_file_count,
        "/",
        total_downloads,
        "objects",
    )

    return uid, local_path


def load_objects(uids: List[str], download_processes: int = 1) -> Dict[str, str]:
    """Return the path to the object files for the given uids.

    If the object is not already downloaded, it will be downloaded.

    Args:
        uids: A list of uids.
        download_processes: The number of processes to use to download the objects.

    Returns:
        A dictionary mapping the object uid to the local path of where the object
        downloaded.
    """
    object_paths = _load_object_paths()
    out = {}
    if download_processes == 1:
        uids_to_download = []
        for uid in uids:
            if uid.endswith(".glb"):
                uid = uid[:-4]
            if uid not in object_paths:
                warnings.warn(f"Could not find object with uid {uid}. Skipping it.")
                continue
            object_path = object_paths[uid]
            local_path = _ensure_local_object(uid, object_path)
            if os.path.exists(local_path):
                out[uid] = local_path
                continue
            uids_to_download.append((uid, object_path))
        if len(uids_to_download) == 0:
            return out
        start_file_count = len(
            glob.glob(os.path.join(OBJECTS_BASE_PATH, "*", "*.glb"))
        )
        for uid, object_path in uids_to_download:
            uid, local_path = _download_object(
                uid, object_path, len(uids_to_download), start_file_count
            )
            out[uid] = local_path
    else:
        args = []
        for uid in uids:
            if uid.endswith(".glb"):
                uid = uid[:-4]
            if uid not in object_paths:
                warnings.warn(f"Could not find object with uid {uid}. Skipping it.")
                continue
            object_path = object_paths[uid]
            local_path = _ensure_local_object(uid, object_path)
            if not os.path.exists(local_path):
                args.append((uid, object_paths[uid]))
            else:
                out[uid] = local_path
        if len(args) == 0:
            return out
        print(
            f"starting download of {len(args)} objects with {download_processes} processes"
        )
        start_file_count = len(
            glob.glob(os.path.join(OBJECTS_BASE_PATH, "*", "*.glb"))
        )
        args = [(*arg, len(args), start_file_count) for arg in args]
        with multiprocessing.Pool(download_processes) as pool:
            r = pool.starmap(_download_object, args)
            for uid, local_path in r:
                out[uid] = local_path
    return out


def load_lvis_annotations() -> Dict[str, List[str]]:
    """Load the LVIS annotations.

    If the annotations are not already downloaded, they will be downloaded.

    Returns:
        A dictionary mapping the LVIS category to the list of uids in that category.
    """
    hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/lvis-annotations.json.gz"
    local_path = os.path.join(_VERSIONED_PATH, "lvis-annotations.json.gz")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        _urlretrieve_with_retry(hf_url, local_path)
    with gzip.open(local_path, "rb") as f:
        lvis_annotations = json.load(f)
    return lvis_annotations


if __name__ == "__main__":
    object_paths = _load_object_paths()
    uids = [k for k, v in object_paths.items()][
        0:10000
    ]
    load_annotations(uids)
    print(f"Loaded {len(uids)} uids")
    objects = load_objects(uids, download_processes=10)
    print(f"Loaded {len(objects)} objects")
