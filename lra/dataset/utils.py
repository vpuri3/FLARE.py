#
import os
import torch
from pathlib import Path
from torch.utils.data import Dataset

import pickle
from lra.dataset.lra import build_lra_filename, scan_samples_for_metadata, LRADataset
from lra.dataset.mta import generate_mta_classification_dataset, save_mta_dataset, load_mta_dataset, MTADataset
from lra.dataset.strassen import FunctCompDataset, BinaryRelationCompDataset, QuotientBinaryRelationCompDataset, Match3Dataset
from lra.dataset.sudoku import SudokuDataset, SudokuProcessConfig, load_or_prepare_split

__all__ = [
    'load_dataset',
]

#======================================================================#
# load_dataset entrypoint
#======================================================================#
def load_dataset(task: str, DATADIR_BASE: str, GLOBAL_RANK: int = 0):
    """Return train, val datasets and metadata for LRA and MTA tasks."""

    LRA_TASKS = ['listops', 'text', 'retrieval', 'image', 'pathfinder32', 'pathfinder128']
    MTA_TASKS = ['mta-toy',]
    STRASSEN_TASKS = ['function_composition', 'binary_relation_composition', 'quotient_binary_relation_composition', 'match2', 'match3',]
    PUZZLE_TASKS = ['sudoku', 'maze']

    if task in STRASSEN_TASKS:
        STRASSEN_CLASS_DICT = dict(
            function_composition=FunctCompDataset,
            binary_relation_composition=BinaryRelationCompDataset,
            quotient_binary_relation_composition=QuotientBinaryRelationCompDataset,
            match3=Match3Dataset,
        )

        DATA_ROOT = os.path.join(DATADIR_BASE, f'Strassen')
        
        # Create full dataset
        full_dataset = STRASSEN_CLASS_DICT[task](DATA_ROOT=DATA_ROOT, GLOBAL_RANK=GLOBAL_RANK)
        
        # Split into 90% train and 10% val
        total_size = len(full_dataset)
        split_idx = int(total_size * 0.9)
        
        # Create train and val subsets
        class SubsetDataset(Dataset):
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]
        
        train_indices = list(range(split_idx))
        val_indices = list(range(split_idx, total_size))
        
        train_ds = SubsetDataset(full_dataset, train_indices)
        val_ds = SubsetDataset(full_dataset, val_indices)
        
        # Compute metadata - scan entire dataset to get accurate max_vocab
        max_vocab = 0
        max_length = 0
        
        print(f"Computing metadata from entire dataset ({len(full_dataset)} samples)...")
        for i in range(len(full_dataset)):
            sample = full_dataset[i]
            input_ids = sample['input_ids']
            max_vocab = max(max_vocab, input_ids.max().item())
            max_length = max(max_length, len(input_ids))
        
        vocab_size = max_vocab + 1  # V = max(input value) + 1, this is the number of input classes
        pad_id = vocab_size  # pad_id = V (which is V+1 from 0-indexed perspective, but we use V as pad_id)
        
        print(f"Computed vocab_size={vocab_size}, pad_id={pad_id}, max_length={max_length}")

        metadata = dict(
            task=task,
            num_labels=2,
            vocab_size=vocab_size,
            pad_id=pad_id,
            ignore_index=-100,      # default ignore index for cross entropy loss
            max_length=max_length,
        )

    elif task == 'sudoku':
        #======================================================================#
        # Sodoku extreme
        #======================================================================#
        DATA_ROOT = os.path.join(DATADIR_BASE, f'Sudoku')

        config = SudokuProcessConfig()

        train_inputs, train_labels, _ = load_or_prepare_split(DATA_ROOT, 'train', config)
        val_inputs, val_labels, _ = load_or_prepare_split(DATA_ROOT, 'test', config)

        train_ds = SudokuDataset(train_inputs, train_labels)
        val_ds = SudokuDataset(val_inputs, val_labels)

        metadata = dict(
            task=task,
            max_length = int(train_inputs.shape[1]),
            vocab_size = int(max(train_inputs.max(), val_inputs.max()) + 1),
            num_labels = int(max(train_labels.max(), val_labels.max()) + 1),
            pad_id=0,
        )

    elif task in MTA_TASKS:
        #======================================================================#
        # Multi-Token Attention toy task
        #======================================================================#
        DATA_ROOT = os.path.join(DATADIR_BASE, 'MTA_toy')

        # Check if cached dataset exists
        train_npz = Path(DATA_ROOT) / 'train.npz'
        val_npz = Path(DATA_ROOT) / 'dev.npz'

        if train_npz.exists() and val_npz.exists():
            print(f"Loading cached MTA dataset from {DATA_ROOT}")
            train_inputs, train_labels, train_meta = load_mta_dataset(DATA_ROOT, 'train')
            val_inputs, val_labels, val_meta = load_mta_dataset(DATA_ROOT, 'dev')
            metadata = train_meta  # Use train metadata as reference
        else:
            print(f"Generating MTA dataset (not found in {DATA_ROOT})")
            # Generate datasets
            train_inputs, train_labels, metadata = generate_mta_classification_dataset(
                num_blocks=50,
                block_length=5,
                query_length=2,
                num_samples=1_000_000,
                alphabet_size=26,
                seed=42,
            )

            val_inputs, val_labels, _ = generate_mta_classification_dataset(
                num_blocks=50,
                block_length=5,
                query_length=2,
                num_samples=1_000,
                alphabet_size=26,
                seed=43,
            )

            # Save to disk
            save_mta_dataset(DATA_ROOT, train_inputs, train_labels, metadata, 'train')
            save_mta_dataset(DATA_ROOT, val_inputs, val_labels, metadata, 'dev')

        # Create dataset objects
        train_ds = MTADataset(train_inputs, train_labels)
        val_ds = MTADataset(val_inputs, val_labels)

        # Package metadata
        metadata_out = dict(
            task=task,
            num_labels=metadata['num_blocks'],
            vocab_size=metadata['vocab_size'],
            max_length=metadata['seq_len'],
        )
        
    elif task in LRA_TASKS:
        #======================================================================#
        # Long Range Arena
        #======================================================================#
        DATA_ROOT = os.path.join(DATADIR_BASE, 'LongRangeArena')
        split_map = dict(train='train', val='dev')

        if not os.path.exists(DATA_ROOT):
            raise ValueError(f"Data root {DATA_ROOT} does not exist. Please download the dataset from https://www.kaggle.com/datasets/a24998667/long-range-arena-processed and unzip it to {DATA_ROOT}")

        # Load pickles (lists of dicts)
        with open(build_lra_filename(task, split_map['train'], DATA_ROOT), 'rb') as f:
            train_list = pickle.load(f)
        with open(build_lra_filename(task, split_map['val'], DATA_ROOT), 'rb') as f:
            val_list = pickle.load(f)

        t_max_tok, t_max_lab, t_len0 = scan_samples_for_metadata(train_list, task)
        v_max_tok, v_max_lab, v_len0 = scan_samples_for_metadata(val_list, task)

        vocab_size = max(t_max_tok, v_max_tok) + 1
        num_labels = int(max(t_max_lab, v_max_lab) + 1)
        seq_len = max(t_len0, v_len0)

        train_ds = LRADataset(train_list, task=task, seq_len=seq_len)
        val_ds = LRADataset(val_list, task=task, seq_len=seq_len)

        metadata = dict(
            task=task,
            num_labels=num_labels,
            vocab_size=vocab_size,
            max_length=seq_len,
        )

    else:
        #======================================================================#
        # Unknown task
        #======================================================================#
        raise ValueError(f"Unknown task: {task}")

    assert metadata['num_labels'] >= 2, f"Number of labels must be at least 2 for classification tasks. Got {metadata['num_labels']} for task {task}"
    metadata['binary_classification'] = metadata['num_labels'] == 2

    return train_ds, val_ds, metadata

#======================================================================#
#