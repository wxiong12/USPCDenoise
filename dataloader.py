import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import torch
from torch.utils.data import DataLoader, Dataset, Subset


SplitLoaders = Dict[str, Dict[str, DataLoader]]

# category -> (train_val file, test file)
CATEGORIES = {
    "data": ("train_val_data.h5", "test_data.h5"),  # terrain, 3072 points
    "unsupervised": ("train_val_unsupervised.h5", "test_unsupervised.h5"),  # foundation, 2048 points
}


class H5PointCloudDataset(Dataset):
    """Lazy HDF5 point cloud dataset for safe use with DataLoader workers."""

    def __init__(self, h5_path: Union[str, Path], dataset_key: Optional[str] = None) -> None:
        self.h5_path = Path(h5_path)
        self.dataset_key = dataset_key or self._find_dataset_key()
        self._file: h5py.File | None = None
        self._dataset = None

        with h5py.File(self.h5_path, "r") as h5_file:
            self.length = int(h5_file[self.dataset_key].shape[0])

    def _find_dataset_key(self) -> str:
        with h5py.File(self.h5_path, "r") as h5_file:
            dataset_keys = [key for key, value in h5_file.items() if isinstance(value, h5py.Dataset)]

        if len(dataset_keys) != 1:
            raise ValueError(
                f"Expected exactly one top-level dataset in {self.h5_path}, "
                f"found {dataset_keys}. Please pass dataset_key explicitly."
            )
        return dataset_keys[0]

    @property
    def dataset(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
            self._dataset = self._file[self.dataset_key]
        return self._dataset

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        points = torch.from_numpy(self.dataset[index]).float()
        return {
            "points": points,
            "index": index,
            "source": self.h5_path.name,
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_dataset"] = None
        return state

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._dataset = None


def split_indices(
    length: int,
    train_ratio: float = 0.9,
    seed: int = 2025,
) -> Tuple[List[int], List[int]]:
    """Split train_val into train / val only. Test comes from test_*.h5."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(length, generator=generator).tolist()
    train_end = round(length * train_ratio)
    return indices[:train_end], indices[train_end:]


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def build_dataloaders(
    data_dir: Union[str, Path] = ".",
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 2025,
    train_ratio: float = 0.9,
    pin_memory: Optional[bool] = None,
) -> SplitLoaders:
    """
    Build DataLoaders from USPCDenoise HDF5 files by category.

    - train_val_*.h5 -> split into train / val only (default 0.9 / 0.1, seed=2025)
    - test_*.h5      -> entire file as test
    """
    data_path = Path(data_dir)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loaders: SplitLoaders = {}

    for category, (train_val_name, test_name) in CATEGORIES.items():
        train_val_path = data_path / train_val_name
        test_path = data_path / test_name

        if not train_val_path.exists() and not test_path.exists():
            continue

        split_loaders: Dict[str, DataLoader] = {}

        if train_val_path.exists():
            train_val_dataset = H5PointCloudDataset(train_val_path)
            train_idx, val_idx = split_indices(
                len(train_val_dataset), train_ratio=train_ratio, seed=seed
            )
            split_loaders["train"] = make_loader(
                Subset(train_val_dataset, train_idx), batch_size, True, num_workers, pin_memory
            )
            split_loaders["val"] = make_loader(
                Subset(train_val_dataset, val_idx), batch_size, False, num_workers, pin_memory
            )

        if test_path.exists():
            test_dataset = H5PointCloudDataset(test_path)
            split_loaders["test"] = make_loader(
                test_dataset, batch_size, False, num_workers, pin_memory
            )

        loaders[category] = split_loaders

    if not loaders:
        raise FileNotFoundError(
            f"No category HDF5 files found in {data_path.resolve()}. "
            f"Expected train_val_*.h5 / test_*.h5 pairs."
        )

    return loaders


def describe_loaders(loaders: SplitLoaders) -> None:
    for name, split_loaders in loaders.items():
        print(name)
        for split, loader in split_loaders.items():
            dataset = loader.dataset
            first_batch = next(iter(loader))
            points = first_batch["points"]
            print(
                f"  {split:<5} samples={len(dataset):>4} "
                f"batches={len(loader):>3} batch_points_shape={tuple(points.shape)}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PyTorch DataLoaders for USPCDenoise HDF5 files.")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Directory containing .h5 files.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for all DataLoaders.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for train/val split.")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train ratio within train_val_*.h5.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaders = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_ratio=args.train_ratio,
    )
    describe_loaders(loaders)


if __name__ == "__main__":
    main()
