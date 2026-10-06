"""Official journal UCR splits and an explicit NPZ route for other sequences."""

import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np


UCR_DATASETS = ("FacesUCR", "FaceAll")
UCR_RECORDS = {"FacesUCR": "11191065", "FaceAll": "11191011"}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_ts(path):
    """Numeric, non-timestamp .ts with equal channel lengths within each sample."""
    values, labels, in_data = [], [], False
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not in_data:
            if line.lower().startswith("@timestamps") and line.lower().split()[1] != "false":
                raise ValueError("Timestamped .ts is unsupported; provide packed NPZ sequences")
            in_data = line.lower() == "@data"
            continue
        parts = line.split(":")
        if len(parts) < 2:
            raise ValueError("Expected channels:class label in .ts file")
        channels = [np.array([float(v) for v in part.split(",")], dtype=np.float64) for part in parts[:-1]]
        if not channels or len({len(c) for c in channels}) != 1 or len(channels[0]) == 0:
            raise ValueError("Empty sequence or unequal channel lengths")
        sequence = np.stack(channels, axis=1)
        if not np.isfinite(sequence).all():
            raise ValueError("Missing/nonfinite frames require explicit preprocessing")
        values.append(sequence)
        labels.append(parts[-1].strip())
    if not values:
        raise ValueError(f"No sequences in {path}")
    return values, np.array(labels)


def fetch_ucr(name, root):
    if name not in UCR_DATASETS:
        raise ValueError("Use FacesUCR/FaceAll or --dataset-file for another journal dataset")
    folder = Path(root) / name
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / f"{name}_{split}.ts" for split in ("TRAIN", "TEST")]
    url = f"https://zenodo.org/records/{UCR_RECORDS[name]}"
    for path in paths:
        if not path.is_file():
            temporary = path.with_suffix(".ts.part")
            request = Request(url + f"/files/{path.name}?download=1",
                              headers={"User-Agent": "FlashOPW-research/0.1"})
            with urlopen(request, timeout=90) as response, temporary.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            temporary.replace(path)
    return paths, dict(dataset=name, source_url=url,
                       files={path.name: file_sha256(path) for path in paths},
                       preprocessing="archive values unchanged; no resampling or feature normalization")


def _npz_sequences(saved, split):
    if f"{split}_x" in saved:
        array = saved[f"{split}_x"]
        if array.ndim == 2:
            array = array[:, :, None]
        if array.ndim != 3:
            raise ValueError("train_x/test_x must be (samples, frames, features) or (samples, frames)")
        return [np.asarray(value, dtype=np.float64) for value in array]
    values, offsets = saved[f"{split}_values"], saved[f"{split}_offsets"]
    if (values.ndim != 2 or offsets.ndim != 1 or len(offsets) < 2
            or not np.issubdtype(offsets.dtype, np.integer) or offsets[0] != 0
            or offsets[-1] != len(values) or (np.diff(offsets) <= 0).any()):
        raise ValueError("Packed sequences need (total frames, features) values and increasing integer offsets")
    return [np.asarray(values[start:end], dtype=np.float64)
            for start, end in zip(offsets[:-1], offsets[1:])]


def load_sequences(name, root, dataset_file=None):
    if dataset_file is None:
        paths, provenance = fetch_ucr(name, root)
        train, train_labels = read_ts(paths[0])
        test, test_labels = read_ts(paths[1])
    else:
        with np.load(dataset_file, allow_pickle=False) as saved:
            train, test = _npz_sequences(saved, "train"), _npz_sequences(saved, "test")
            train_labels, test_labels = saved["train_labels"].copy(), saved["test_labels"].copy()
        provenance = dict(dataset=name, source_file=str(Path(dataset_file).resolve()),
                          sha256=file_sha256(dataset_file), preprocessing="provided sequences unchanged")
    if (not train or not test or train_labels.shape != (len(train),)
            or test_labels.shape != (len(test),)):
        raise ValueError("Nonempty sequences and matching one-dimensional labels required")
    dimensions = {value.shape[1] for value in train + test}
    if len(dimensions) != 1 or min(dimensions) < 1:
        raise ValueError("All sequences need the same positive feature dimension")
    if any(len(value) < 1 or not np.isfinite(value).all() for value in train + test):
        raise ValueError("Empty or nonfinite sequences")
    provenance.update(original_train=len(train), original_test=len(test),
                      classes=len(np.unique(train_labels)), features=next(iter(dimensions)))
    return train, train_labels, test, test_labels, provenance


def balanced_subset(labels, limit, seed):
    """Round-robin random class samples, returned in original split order."""
    if limit == 0 or limit >= len(labels):
        return np.arange(len(labels))
    classes = np.unique(labels)
    if limit < len(classes):
        raise ValueError("Subset limit must include at least one sample of every class")
    generator = np.random.default_rng(seed)
    pools = [generator.permutation(np.flatnonzero(labels == label)).tolist() for label in classes]
    selected = []
    while len(selected) < limit:
        for pool in pools:
            if pool and len(selected) < limit:
                selected.append(pool.pop())
    return np.array(sorted(selected), dtype=np.int64)
