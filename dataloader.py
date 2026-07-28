import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import torch
from torch.utils.data import DataLoader, Dataset, Subset


SplitLoaders = Dict[str, Dict[str, DataLoader]]


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
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(length, generator=generator).tolist()

    train_end = int(length * train_ratio)
    val_end = train_end + int(length * val_ratio)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


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
    seed: int = 42,
    pin_memory: Optional[bool] = None,
) -> SplitLoaders:
    """
    Build DataLoaders from USPCDenoise HDF5 files.

    Files starting with "train" are split into train/val/test as 0.8/0.1/0.1.
    Files starting with "test" are used only as test sets.
    Loaders are grouped by file stem to avoid mixing point clouds with different
    numbers of points in the same batch.
    """

    data_path = Path(data_dir)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loaders: SplitLoaders = {}

    for h5_path in sorted(data_path.glob("train*.h5")):
        dataset = H5PointCloudDataset(h5_path)
        train_idx, val_idx, test_idx = split_indices(len(dataset), seed=seed)

        loaders[h5_path.stem] = {
            "train": make_loader(Subset(dataset, train_idx), batch_size, True, num_workers, pin_memory),
            "val": make_loader(Subset(dataset, val_idx), batch_size, False, num_workers, pin_memory),
            "test": make_loader(Subset(dataset, test_idx), batch_size, False, num_workers, pin_memory),
        }

    for h5_path in sorted(data_path.glob("test*.h5")):
        dataset = H5PointCloudDataset(h5_path)
        loaders[h5_path.stem] = {
            "test": make_loader(dataset, batch_size, False, num_workers, pin_memory),
        }

    if not loaders:
        raise FileNotFoundError(f"No train*.h5 or test*.h5 files found in {data_path.resolve()}")

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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val/test split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaders = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    describe_loaders(loaders)


if __name__ == "__main__":
    main()