"""Domain-aware datasets and pair construction for MACAD.

Each domain is expected to provide a pre-split ``train`` and ``test`` sequence.
For a source domain they are used as source-train and source-validation,
respectively.  For a target domain they are used for unlabeled adaptation and
final evaluation, respectively.
"""

import ast
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.stattools import adfuller
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.augmentations import Injector


DATASET_VERSION = 10


def _dataset_name(path):
    path_lower = str(path).lower()
    for name in ("smd", "msl", "smap", "boiler"):
        if name in path_lower:
            return name
    raise ValueError(f"Cannot infer dataset type from path: {path}")


def get_dataset(args, domain_type, split_type):
    """Build one source/target dataset without crossing split boundaries."""
    if domain_type not in {"source", "target"}:
        raise ValueError("domain_type must be 'source' or 'target'")
    if split_type not in {"train", "val", "test"}:
        raise ValueError("split_type must be 'train', 'val', or 'test'")

    root = args.path_src if domain_type == "source" else args.path_trg
    subject_id = args.id_src if domain_type == "source" else args.id_trg
    name = _dataset_name(root)
    default_windows = {"msl": 100, "smap": 100, "smd": 100, "boiler": 100}
    dataset_cls = {
        "msl": MSLDataset,
        "smap": SMAPDataset,
        "smd": SMDDataset,
        "boiler": BoilerDataset,
    }[name]
    return dataset_cls(
        root,
        subject_id=subject_id,
        split_type=split_type,
        domain_type=domain_type,
        window_size=getattr(args, "window_size", default_windows[name]),
    )


class MACADDataset(Dataset):
    """Shared leakage-free loading and MACAD pair mining."""

    dataset_name = None
    def __init__(
        self,
        root_dir,
        subject_id,
        split_type="train",
        domain_type="source",
        verbose=True,
        window_size=100,
    ):
        self.root_dir = str(root_dir)
        self.subject_id = str(subject_id)
        self.split_type = split_type
        self.domain_type = domain_type
        self.verbose = verbose
        self.window_size = int(window_size)

        self._load_sequence()
        if len(self.sequence) < 4:
            raise ValueError(
                f"{self.dataset_name}/{self.subject_id}/{split_type} has only "
                f"{len(self.sequence)} windows; at least four are required"
            )

        # Pairs and augmentations are needed only by the adaptation split.
        # Validation/test datasets remain lightweight prediction datasets.
        self.has_pairs = split_type == "train"
        self.cache_path = None
        if self.has_pairs:
            cache_dir = self._dataset_root() / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_name = (
                f"macad_v{DATASET_VERSION}_{self.dataset_name}_{self.subject_id}_"
                f"{domain_type}_{split_type}_w{self.window_size}.pkl"
            )
            self.cache_path = cache_dir / cache_name
            if not self._load_cache():
                self._build_pairs()
                self._save_cache()

        if self.verbose:
            print(
                f"[{self.dataset_name.upper()}:{domain_type}:{split_type}] "
                f"{len(self.sequence)} windows"
                + (f", epsilon={self.epsilon}, delta={self.delta}" if self.has_pairs else "")
            )

    def _dataset_root(self):
        root = Path(self.root_dir)
        if self.dataset_name == "smd" and root.name.lower() in {
            "train", "test", "test_label", "train_label"
        }:
            return root.parent
        return root

    def _read_raw_train(self):
        raise NotImplementedError

    def _read_raw_test(self):
        raise NotImplementedError

    def _domain_csv_path(self, split):
        """Resolve the Boiler-style ``<domain>/<split>.csv`` layout."""
        root = self._dataset_root()
        subject = self.subject_id.removesuffix(".txt")
        candidates = [
            root / subject / f"{split}.csv",
            root / f"machine-{subject}" / f"{split}.csv",
        ]
        if subject.startswith("machine-"):
            candidates.append(root / subject.removeprefix("machine-") / f"{split}.csv")
        return next((path for path in candidates if path.is_file()), None)

    def _read_domain_csv(self, split):
        """Read a shared Boiler-style CSV, or return ``None`` for legacy data."""
        path = self._domain_csv_path(split)
        if path is None:
            return None
        frame = pd.read_csv(path)
        if frame.shape[1] < 4:
            raise ValueError(
                f"{path} must contain two metadata columns, at least one feature, "
                "and one label column"
            )
        features = frame.iloc[:, 2:-1].apply(pd.to_numeric, errors="raise")
        labels = pd.to_numeric(frame.iloc[:, -1], errors="raise")
        return features.to_numpy(dtype=np.float32), labels.to_numpy(dtype=np.int64)

    def _load_sequence(self):
        raw_train, train_labels = self._read_raw_train()
        raw_train = self._clean_array(raw_train)
        train_labels = self._clean_labels(train_labels, len(raw_train))
        fit_data = raw_train

        if self.domain_type == "source":
            if self.split_type == "test":
                raise ValueError(
                    "Source test.csv is assigned to source validation; request "
                    "split_type='val' instead."
                )
            if self.split_type == "train":
                raw, labels = raw_train, train_labels
            else:
                raw, labels = self._read_raw_test()
                raw = self._clean_array(raw)
                labels = self._clean_labels(labels, len(raw))
        else:
            # Target adaptation is strictly unlabeled even if a dataset file
            # happens to carry an annotation column (for example Boiler).
            adaptation_labels = np.zeros(len(raw_train), dtype=np.int64)
            if self.split_type in {"train", "val"}:
                # Target train is unlabeled adaptation data. ``val`` remains an
                # alias for compatibility; training uses the same dataset object.
                raw, labels = raw_train, adaptation_labels
            else:
                raw, labels = self._read_raw_test()
                raw = self._clean_array(raw)
                labels = self._clean_labels(labels, len(raw))

        # Source statistics use source-train only; target statistics use the
        # official target train only. Target test never affects preprocessing.
        self.mean = fit_data.mean(axis=0)
        self.std = fit_data.std(axis=0)
        self.std[self.std == 0.0] = 1.0

        normalized = (raw - self.mean) / self.std
        self.sequence, self.label, self.point_label = self.convert_to_windows(
            normalized, labels, self.window_size
        )

    @staticmethod
    def _clean_array(values):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        return np.nan_to_num(values, copy=False)

    @staticmethod
    def _clean_labels(labels, expected_length):
        labels = np.asarray(labels).reshape(-1).astype(np.int64)
        if len(labels) != expected_length:
            raise ValueError(
                f"Sequence/label length mismatch: {expected_length} != {len(labels)}"
            )
        return labels

    @staticmethod
    def convert_to_windows(sequence, labels, window_size, stride=1):
        count = (len(sequence) - window_size) // stride + 1
        if count <= 0:
            raise ValueError("Sequence is shorter than window_size")
        windows = np.stack(
            [sequence[i * stride:i * stride + window_size] for i in range(count)]
        )
        window_labels = np.fromiter(
            (
                int(labels[i * stride:i * stride + window_size].any())
                for i in range(count)
            ),
            dtype=np.int64,
            count=count,
        )
        # A window score is assigned to its final timestamp for point-wise
        # evaluation. This avoids expanding one anomalous point to every
        # overlapping window as the training label intentionally does.
        point_labels = labels[window_size - 1::stride].astype(np.int64, copy=False)
        return windows, window_labels, point_labels

    def _load_cache(self):
        if not self.cache_path.exists():
            return False
        try:
            with self.cache_path.open("rb") as handle:
                cache = pickle.load(handle)
            if cache.get("version") != DATASET_VERSION or cache.get("length") != len(self):
                return False
            self.adf_p_values = cache["adf_p_values"]
            self.epsilon = cache["epsilon"]
            self.delta = cache["delta"]
            self.fixed_contextual_pairs = cache["fixed_contextual_pairs"]
            self.pos_temporal_candidates = cache["pos_temporal_candidates"]
            self.neg_temporal_candidates = cache["neg_temporal_candidates"]
            return True
        except (KeyError, OSError, pickle.UnpicklingError, EOFError):
            return False

    def _save_cache(self):
        cache = {
            "version": DATASET_VERSION,
            "length": len(self),
            "adf_p_values": self.adf_p_values,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "fixed_contextual_pairs": self.fixed_contextual_pairs,
            "pos_temporal_candidates": self.pos_temporal_candidates,
            "neg_temporal_candidates": self.neg_temporal_candidates,
        }
        with self.cache_path.open("wb") as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _build_pairs(self):
        self.adf_p_values = self._compute_adf_values()
        mean_p = float(np.mean(self.adf_p_values))
        self.epsilon = max(50, 250 - int(200 * mean_p))
        # A strict >5*epsilon negative must exist even for a center window.
        # Short synthetic/test sequences therefore cap the ADF-derived range.
        max_strict_epsilon = (len(self) // 2 - 1) // 5
        if max_strict_epsilon < 1:
            raise ValueError("At least 12 training windows are required for temporal pairs")
        self.epsilon = min(self.epsilon, max_strict_epsilon)
        self.delta = self.epsilon * 5

        # Contextual pairs are fixed first. Temporal pools explicitly remove
        # both selected contextual indices.
        self.fixed_contextual_pairs = self._compute_contextual_pairs()
        self.pos_temporal_candidates, self.neg_temporal_candidates = (
            self._compute_temporal_candidates()
        )

    def _compute_adf_values(self):
        values = np.zeros(len(self), dtype=np.float32)
        iterator = tqdm(
            range(len(self)),
            desc=f"[{self.dataset_name.upper()}] ADF",
            disable=not self.verbose,
        )
        for idx in iterator:
            p_values = []
            for column in self.sequence[idx].T:
                if len(column) < 10 or np.std(column) < 1e-8:
                    p_values.append(0.01)
                    continue
                try:
                    p_value = adfuller(column, maxlag=1, autolag=None)[1]
                except (ValueError, np.linalg.LinAlgError):
                    p_value = 0.01
                p_values.append(p_value if np.isfinite(p_value) and p_value <= 1 else 0.01)
            values[idx] = np.mean(p_values)
        return values

    def _compute_contextual_pairs(self):
        if self.domain_type == "source":
            return self._compute_source_contextual_pairs()
        return self._compute_target_contextual_pairs()

    def _compute_source_contextual_pairs(self):
        """Use same-label positives and different-label negatives (Eq. 2)."""
        pairs = {}
        indices = np.arange(len(self), dtype=np.int64)
        missing_negative_class = len(np.unique(self.label)) < 2
        if missing_negative_class:
            warnings.warn(
                f"{self.dataset_name}/{self.subject_id} source-train contains one class; "
                "contextual negatives will use pseudo-anomaly transformations because "
                "the chronological split provides no different-label candidate.",
                RuntimeWarning,
            )
        for idx, label in enumerate(self.label):
            positives = indices[(self.label == label) & (indices != idx)]
            negatives = indices[self.label != label]
            if not len(positives):
                raise ValueError(
                    "Source-train must contain at least two windows in each anchor class"
                )
            pairs[idx] = (
                int(positives[idx % len(positives)]),
                int(negatives[idx % len(negatives)]) if len(negatives) else -1,
            )
        return pairs

    def _compute_target_contextual_pairs(self):
        """Find global FFT-magnitude nearest/farthest windows (Eq. 3)."""
        spectra = np.abs(np.fft.fft(self.sequence, axis=1)).reshape(len(self), -1)
        spectra = spectra.astype(np.float32, copy=False)
        if torch.cuda.is_available() and len(self) > 5000:
            return self._compute_target_contextual_pairs_cuda(spectra)

        squared_norms = np.sum(spectra * spectra, axis=1)
        pairs = {}
        chunk_size = 256
        for start in range(0, len(self), chunk_size):
            end = min(start + chunk_size, len(self))
            distances = (
                squared_norms[start:end, None] + squared_norms[None, :]
                - 2.0 * spectra[start:end] @ spectra.T
            )
            np.maximum(distances, 0.0, out=distances)
            row_indices = np.arange(end - start)
            self_indices = np.arange(start, end)
            distances[row_indices, self_indices] = np.inf
            nearest = np.argmin(distances, axis=1)
            distances[row_indices, self_indices] = -np.inf
            farthest = np.argmax(distances, axis=1)
            for offset, idx in enumerate(range(start, end)):
                pairs[idx] = (int(nearest[offset]), int(farthest[offset]))
        return pairs

    @staticmethod
    def _compute_target_contextual_pairs_cuda(spectra):
        """Exact chunked global search accelerated on CUDA for long domains."""
        device = torch.device("cuda")
        values = torch.as_tensor(spectra, device=device)
        squared_norms = torch.sum(values * values, dim=1)
        pairs = {}
        chunk_size = 512
        with torch.no_grad():
            for start in range(0, len(values), chunk_size):
                end = min(start + chunk_size, len(values))
                distances = (
                    squared_norms[start:end, None] + squared_norms[None, :]
                    - 2.0 * values[start:end] @ values.T
                ).clamp_min_(0.0)
                row_indices = torch.arange(end - start, device=device)
                self_indices = torch.arange(start, end, device=device)
                distances[row_indices, self_indices] = torch.inf
                nearest = torch.argmin(distances, dim=1)
                distances[row_indices, self_indices] = -torch.inf
                farthest = torch.argmax(distances, dim=1)
                nearest = nearest.cpu().tolist()
                farthest = farthest.cpu().tolist()
                for offset, idx in enumerate(range(start, end)):
                    pairs[idx] = (nearest[offset], farthest[offset])
                del distances
        del values, squared_norms
        return pairs

    def _compute_temporal_candidates(self):
        positives, negatives = {}, {}
        all_indices = np.arange(len(self), dtype=np.int64)
        for idx in range(len(self)):
            contextual = set(self.fixed_contextual_pairs[idx])

            start = max(0, idx - self.epsilon)
            end = min(len(self), idx + self.epsilon + 1)
            positive = np.arange(start, end, dtype=np.int64)
            positive = np.asarray(
                [item for item in positive if item != idx and item not in contextual],
                dtype=np.int64,
            )

            far_start = max(0, idx - self.delta)
            far_end = min(len(self), idx + self.delta + 1)
            negative = all_indices[(all_indices < far_start) | (all_indices >= far_end)]
            negative = np.asarray(
                [item for item in negative if item != idx and item not in contextual],
                dtype=np.int64,
            )

            if not len(positive) or not len(negative):
                raise ValueError(
                    "Unable to create strict temporal pairs after excluding contextual pairs"
                )

            positives[idx] = positive
            negatives[idx] = negative
        return positives, negatives

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):
        anchor = self.sequence[index]
        result = {
            "sequence": torch.as_tensor(anchor, dtype=torch.float32),
            "label": torch.tensor([self.label[index]], dtype=torch.long),
            "point_label": torch.tensor([self.point_label[index]], dtype=torch.long),
        }
        if not self.has_pairs:
            return result

        pos_context_idx, neg_context_idx = self.fixed_contextual_pairs[index]
        pos_temporal_idx = int(np.random.choice(self.pos_temporal_candidates[index]))
        neg_temporal_idx = int(np.random.choice(self.neg_temporal_candidates[index]))

        std = np.std(anchor, axis=0)
        positive_transform = anchor + np.random.normal(
            0, 0.1 * std, size=anchor.shape
        )
        negative_transform = get_injector(anchor, self.mean, self.std)
        negative_context = (
            self.sequence[neg_context_idx]
            if neg_context_idx >= 0
            else get_injector(anchor, self.mean, self.std)
        )

        positive = np.stack(
            [self.sequence[pos_context_idx], self.sequence[pos_temporal_idx], positive_transform],
            axis=0,
        )
        negative = np.stack(
            [negative_context, self.sequence[neg_temporal_idx], negative_transform],
            axis=0,
        )
        result["positive"] = torch.as_tensor(positive, dtype=torch.float32)
        result["negative"] = torch.as_tensor(negative, dtype=torch.float32)
        return result

    def get_statistic(self):
        return self.mean.copy(), self.std.copy()


class NASADataset(MACADDataset):
    def _read_raw_train(self):
        csv_data = self._read_domain_csv("train")
        if csv_data is not None:
            return csv_data
        data = np.load(self._dataset_root() / "train" / f"{self.subject_id}.npy")
        return data, np.zeros(len(data), dtype=np.int64)

    def _read_raw_test(self):
        csv_data = self._read_domain_csv("test")
        if csv_data is not None:
            return csv_data
        data = np.load(self._dataset_root() / "test" / f"{self.subject_id}.npy")
        metadata = pd.read_csv(self._dataset_root() / "labeled_anomalies.csv")
        rows = metadata[metadata["chan_id"].astype(str) == self.subject_id]
        if rows.empty:
            raise ValueError(f"No anomaly metadata for channel {self.subject_id}")
        labels = np.zeros(len(data), dtype=np.int64)
        for value in rows["anomaly_sequences"]:
            for start, end in ast.literal_eval(value):
                labels[max(0, start):min(len(labels), end + 1)] = 1
        return data, labels


class MSLDataset(NASADataset):
    dataset_name = "msl"


class SMAPDataset(NASADataset):
    dataset_name = "smap"


class SMDDataset(MACADDataset):
    dataset_name = "smd"

    def _smd_path(self, split):
        return self._dataset_root() / split / f"machine-{self.subject_id}.txt"

    def _read_raw_train(self):
        csv_data = self._read_domain_csv("train")
        if csv_data is not None:
            return csv_data
        data = np.loadtxt(self._smd_path("train"), delimiter=",")
        return data, np.zeros(len(data), dtype=np.int64)

    def _read_raw_test(self):
        csv_data = self._read_domain_csv("test")
        if csv_data is not None:
            return csv_data
        data = np.loadtxt(self._smd_path("test"), delimiter=",")
        labels = np.loadtxt(self._smd_path("test_label"), delimiter=",")
        return data, labels


class BoilerDataset(MACADDataset):
    dataset_name = "boiler"

    def _read_raw_train(self):
        data = self._read_domain_csv("train")
        if data is None:
            raise FileNotFoundError(
                f"No train.csv found for Boiler domain {self.subject_id}"
            )
        return data

    def _read_raw_test(self):
        data = self._read_domain_csv("test")
        if data is None:
            raise FileNotFoundError(
                f"No test.csv found for Boiler domain {self.subject_id}"
            )
        return data


def get_injector(window, data_mean, data_std):
    original_scale = (window * data_std) + data_mean
    injected = Injector(original_scale).injected_win
    return (injected - data_mean) / data_std
