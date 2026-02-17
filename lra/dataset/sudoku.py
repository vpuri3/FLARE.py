#
import csv
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:  # pragma: no cover - handled upstream
    raise ImportError(
        "huggingface_hub is required to prepare the Sudoku dataset."
    ) from exc


#======================================================================#
# Dataset configuration
#======================================================================#
@dataclass
class SudokuProcessConfig:
    source_repo: str = "sapientinc/sudoku-extreme"
    min_difficulty: Optional[int] = None
    subsample_size: Optional[int] = 1 # TODO (2, 1000)
    num_aug: int = 1000 # TODO 1000
    seed: Optional[int] = 0

class SudokuDataset(Dataset):
    """Dataset wrapper for Sudoku Extreme puzzles."""

    def __init__(self, inputs: np.ndarray, labels: np.ndarray):
        assert inputs.shape == labels.shape, "Inputs and labels must have identical shapes"
        assert inputs.ndim == 2, "Inputs must be 2D [num_examples, 81]"
        self.inputs = inputs.astype(np.int64, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.seq_len = self.inputs.shape[1]

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        input_ids = torch.from_numpy(self.inputs[idx])
        label_ids = torch.from_numpy(self.labels[idx])
        return dict(input_ids=input_ids, labels=label_ids)

def load_or_prepare_split(
    data_root: str,
    split: str,
    config: SudokuProcessConfig,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load cached split or download & prepare it if missing."""

    split_dir = os.path.join(data_root, split)
    inputs_path = os.path.join(split_dir, "inputs.npy")
    labels_path = os.path.join(split_dir, "labels.npy")
    metadata_path = os.path.join(split_dir, "metadata.json")

    if not (os.path.exists(inputs_path) and os.path.exists(labels_path)):
        _prepare_split(data_root, split, config)

    inputs = np.load(inputs_path, allow_pickle=False).astype(np.int64, copy=False)
    labels = np.load(labels_path, allow_pickle=False).astype(np.int64, copy=False)

    # TODO: Remove after debugging
    if split != 'train':
        inputs = inputs[:10_000]
        labels = labels[:10_000]

    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata.setdefault("num_examples", int(inputs.shape[0]))
    metadata.setdefault("seq_len", int(inputs.shape[1]))
    metadata.setdefault("min_value", int(inputs.min()))
    metadata.setdefault("max_value", int(inputs.max()))

    return inputs, labels, metadata

#======================================================================#
# Helpers
#======================================================================#
def _prepare_split(data_root: str, split: str, config: SudokuProcessConfig):
    rng = _make_rng(config.seed, split)
    print(f"Downloading Sudoku {split} split from {config.source_repo}...")
    csv_path = hf_hub_download(config.source_repo, filename=f"{split}.csv", repo_type="dataset")
    print(f"Downloaded to {csv_path}")

    print(f"Loading and parsing {split} puzzles...")
    puzzles, solutions = _load_boards(csv_path, config)
    print(f"Loaded {len(puzzles)} puzzles")

    if split == "train" and config.subsample_size is not None:
        total = len(puzzles)
        target = min(config.subsample_size, total)
        if target < total:
            indices = rng.choice(total, size=target, replace=False)
            puzzles = [puzzles[i] for i in indices]
            solutions = [solutions[i] for i in indices]

    num_augments = config.num_aug if split == "train" else 0

    inputs_list = []
    labels_list = []

    if num_augments > 0:
        print(f"Generating {num_augments} augmentations per puzzle...")

    for puzzle, solution in zip(puzzles, solutions):
        inputs_list.append(puzzle.reshape(-1).copy())
        labels_list.append(solution.reshape(-1).copy())

        for _ in range(num_augments):
            aug_inp, aug_out = shuffle_sudoku(puzzle, solution, rng)
            inputs_list.append(aug_inp.reshape(-1).copy())
            labels_list.append(aug_out.reshape(-1).copy())

    inputs = np.stack(inputs_list, axis=0).astype(np.int64)
    labels = np.stack(labels_list, axis=0).astype(np.int64)

    split_dir = os.path.join(data_root, split)
    os.makedirs(split_dir, exist_ok=True)

    print(f"Saving {len(inputs)} examples to {split_dir}...")
    np.save(os.path.join(split_dir, "inputs.npy"), inputs)
    np.save(os.path.join(split_dir, "labels.npy"), labels)

    metadata = {
        "num_examples": int(inputs.shape[0]),
        "seq_len": int(inputs.shape[1]),
        "min_value": int(inputs.min()),
        "max_value": int(inputs.max()),
        "num_augments": int(num_augments),
    }

    with open(os.path.join(split_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Successfully prepared {split} split with {len(inputs)} examples")


def _load_boards(csv_path: str, config: SudokuProcessConfig):
    puzzles = []
    solutions = []

    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"CSV file {csv_path} is empty")

        for row in reader:
            if len(row) < 4:
                continue
            _, puzzle_str, solution_str, rating_str = row[:4]

            if config.min_difficulty is not None:
                try:
                    rating = int(rating_str)
                except ValueError:
                    continue
                if rating < config.min_difficulty:
                    continue

            puzzle_board = _decode_board(puzzle_str, blank_char=".")
            solution_board = _decode_board(solution_str, blank_char=None)
            puzzles.append(puzzle_board)
            solutions.append(solution_board)

    if not puzzles:
        raise ValueError("No Sudoku puzzles loaded. Check filtering parameters.")

    return puzzles, solutions

def _decode_board(board_str: str, blank_char: Optional[str]) -> np.ndarray:
    if len(board_str) != 81:
        raise ValueError(f"Sudoku board must have length 81. Got {len(board_str)}")

    if blank_char is not None:
        board_str = board_str.replace(blank_char, "0")

    data = np.frombuffer(board_str.encode("ascii"), dtype=np.uint8).astype(np.int64)
    data = data - ord("0")
    if data.min() < 0 or data.max() > 9:
        raise ValueError("Board values must be in [0, 9]")
    return data.reshape(9, 9)

def _make_rng(seed: Optional[int], split: str) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    split_offset = {"train": 0, "test": 1, "val": 2}.get(split, 3)
    return np.random.default_rng(seed + split_offset)

def shuffle_sudoku(
    board: np.ndarray,
    solution: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random symmetry and digit remapping preserving Sudoku validity."""

    digit_map = np.concatenate(([0], rng.permutation(np.arange(1, 10))))
    transpose_flag = rng.random() < 0.5

    bands = rng.permutation(3)
    row_perm = np.concatenate([b * 3 + rng.permutation(3) for b in bands])

    stacks = rng.permutation(3)
    col_perm = np.concatenate([s * 3 + rng.permutation(3) for s in stacks])

    mapping = np.fromiter(
        (row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)),
        dtype=np.int64,
        count=81,
    )

    def apply_transform(x: np.ndarray) -> np.ndarray:
        x_mat = x.reshape(9, 9)
        if transpose_flag:
            x_mat = x_mat.T
        new_board = x_mat.reshape(-1)[mapping].reshape(9, 9)
        return digit_map[new_board].copy()

    return apply_transform(board), apply_transform(solution)


__all__ = [
    "SudokuDataset",
    "SudokuProcessConfig",
    "load_or_prepare_split",
    "shuffle_sudoku",
]
