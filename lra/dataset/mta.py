#
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pickle
import random
import torch
from torch.utils.data import Dataset

from tqdm import tqdm

#======================================================================#
# MTA toy task
# https://arxiv.org/pdf/2504.00927
#======================================================================#
class MTADataset(Dataset):
    """Dataset wrapper for the synthetic MTA classification task."""

    def __init__(self, input_ids: np.ndarray, labels: np.ndarray):
        assert len(input_ids) == len(labels), "Inputs and labels must have same length"
        self.input_ids = input_ids
        self.labels = labels
        self.seq_len = input_ids.shape[1] if len(input_ids) else 0

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = torch.as_tensor(self.input_ids[idx], dtype=torch.long)
        label = torch.as_tensor(int(self.labels[idx]), dtype=torch.long)
        return dict(input_ids=input_ids, labels=label)

def generate_mta_classification_dataset(
    *,
    num_blocks: int = 50,
    block_length: int = 5,
    query_length: int = 2,
    num_samples: int = 0,
    alphabet_size: int = 26,
    seed: Optional[int] = None,
    sep_token: str = '.',
    query_token: str = '#',
    pad_token: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Generate synthetic classification data for the MTA toy task.

    Task: Given a sequence of blocks and query letters, identify which block contains all query letters.

    Input format: q1 ... qL [#] block1 [DOT] block2 [DOT] ... blockB [PAD]*

    Tokenization:
        - 0-25: letters a-z
        - 26: [DOT] (block separator '.')
        - 27: [#] (query region separator)
        - 28: [PAD] (padding, if used)

    Label: Index (0-based) of the block containing ALL query letters.

    Validation: Each generated sample is checked to ensure exactly one block contains all query letters.

    Args:
        num_blocks: Number of blocks per sample (B, default=50).
        block_length: Characters per block (N, default=5).
        query_length: Number of query letters (L, default=2).
        num_samples: Total number of examples to generate.
        alphabet_size: Size of the alphabet (default=26, excludes separators).
        seed: Optional seed for reproducibility.
        sep_token: Token inserted between blocks (default=".").
        query_token: Token separating query from blocks (default="#").
        pad_token: Optional token for padding the sequence.

    Returns:
        Tuple of (inputs, labels, metadata):
            - inputs: np.ndarray of shape (num_samples, seq_len) with token IDs
            - labels: np.ndarray of shape (num_samples,) with block indices
            - metadata: Dict with vocab mappings and dataset info
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    assert query_length > 0, "Query length must be positive"
    assert block_length >= query_length, "Block length must be >= query length"
    unique_tokens = {chr(ord('a') + i) for i in range(26)}
    assert alphabet_size <= len(unique_tokens), "Alphabet size cannot exceed 26"
    assert sep_token not in unique_tokens and query_token not in unique_tokens, (
        "Separators must not collide with the lowercase alphabet."
    )

    # Build vocabulary mapping
    letters = [chr(ord('a') + i) for i in range(26)]
    used_letters = []
    for letter in letters:
        if letter != sep_token and letter != query_token:
            used_letters.append(letter)
        if len(used_letters) == alphabet_size:
            break
    alphabet = used_letters

    token_to_id = {ch: idx for idx, ch in enumerate(alphabet)}
    next_token_id = len(token_to_id)

    if sep_token not in token_to_id:
        token_to_id[sep_token] = next_token_id
        next_token_id += 1
    if query_token not in token_to_id:
        token_to_id[query_token] = next_token_id
        next_token_id += 1
    if pad_token is not None and pad_token not in token_to_id:
        token_to_id[pad_token] = next_token_id

    id_to_token = {idx: tok for tok, idx in token_to_id.items()}

    all_inputs: List[np.ndarray] = []
    all_labels: List[int] = []
    seq_len = num_blocks * block_length + (num_blocks - 1) + 1 + query_length
    if pad_token is not None:
        seq_len = max(seq_len, num_blocks * block_length + num_blocks + query_length)

    for _ in tqdm(range(num_samples), desc="Generating MTA samples", ncols=80):
        # Choose target block and query letters
        target_block_chars = random.sample(alphabet, block_length)
        query_chars = random.sample(target_block_chars, query_length)

        blocks = []
        target_index = random.randrange(num_blocks)

        for block_idx in range(num_blocks):
            if block_idx == target_index:
                blocks.append(target_block_chars.copy())
            else:
                while True:
                    candidate = random.sample(alphabet, block_length)
                    if not all(char in candidate for char in query_chars):
                        blocks.append(candidate)
                        break

        # Permute within-block characters to avoid bias
        for block in blocks:
            random.shuffle(block)

        # Assemble flat sequence of tokens
        tokens = list(query_chars)
        tokens.append(query_token)
        for block_idx, block in enumerate(blocks):
            tokens.extend(block)
            if block_idx != num_blocks - 1:
                tokens.append(sep_token)

        if pad_token is not None:
            while len(tokens) < seq_len:
                tokens.append(pad_token)

        input_ids = np.array([token_to_id[token] for token in tokens], dtype=np.int64)

        # Validation: verify exactly one block contains all query letters
        sep_id = token_to_id[sep_token]
        query_id = token_to_id[query_token]
        query_letter_ids = {token_to_id[ch] for ch in query_chars}

        # Find query separator position
        query_sep_pos = np.where(input_ids == query_id)[0][0]
        # Extract block region (after query separator)
        block_region = input_ids[query_sep_pos + 1:]

        # Split into blocks
        block_boundaries = [0] + [i + 1 for i, tok_id in enumerate(block_region) if tok_id == sep_id] + [len(block_region)]
        found_target_blocks = []
        for i in range(len(block_boundaries) - 1):
            block_tokens = set(block_region[block_boundaries[i]:block_boundaries[i + 1]])
            if query_letter_ids.issubset(block_tokens):
                found_target_blocks.append(i)

        assert len(found_target_blocks) == 1, f"Expected exactly 1 block with all query letters, found {len(found_target_blocks)}"
        assert found_target_blocks[0] == target_index, f"Target block index mismatch: {found_target_blocks[0]} != {target_index}"

        all_inputs.append(input_ids)
        all_labels.append(target_index)

    metadata = {
        'token_to_id': token_to_id,
        'id_to_token': id_to_token,
        'vocab_size': len(token_to_id),
        'seq_len': seq_len,
        'num_blocks': num_blocks,
        'block_length': block_length,
        'query_length': query_length,
    }

    inputs_array = np.stack(all_inputs, axis=0) if all_inputs else np.zeros((0, seq_len), dtype=np.int64)
    labels_array = np.array(all_labels, dtype=np.int64)

    return inputs_array, labels_array, metadata

def save_mta_dataset(save_dir: str, inputs: np.ndarray, labels: np.ndarray, metadata: Dict, split: str):
    """Save MTA dataset to disk in both text and npz formats."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save npz file with inputs, labels, and metadata
    npz_file = save_path / f"{split}.npz"
    np.savez(
        npz_file,
        inputs=inputs,
        labels=labels,
        **{k: v for k, v in metadata.items() if not isinstance(v, dict)}
    )
    
    # Save metadata separately as pickle (for dict fields)
    meta_file = save_path / f"{split}_metadata.pkl"
    with open(meta_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save human-readable text file
    txt_file = save_path / f"{split}.txt"
    id_to_token = metadata['id_to_token']
    with open(txt_file, 'w') as f:
        f.write(f"# MTA Classification Dataset ({split} split)\n")
        f.write(f"# Format: <input_sequence> | <label>\n")
        f.write(f"# Num samples: {len(inputs)}, Seq len: {inputs.shape[1]}, Vocab size: {metadata['vocab_size']}\n\n")
        
        for inp, lbl in zip(inputs, labels):
            # Convert token IDs to characters
            seq_str = ''.join([id_to_token[int(tok_id)] for tok_id in inp])
            f.write(f"{seq_str} | {lbl}\n")
    
    print(f"Saved {split} dataset to {save_path}")

def load_mta_dataset(save_dir: str, split: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load MTA dataset from disk."""
    save_path = Path(save_dir)
    
    npz_file = save_path / f"{split}.npz"
    meta_file = save_path / f"{split}_metadata.pkl"
    
    if not npz_file.exists() or not meta_file.exists():
        raise FileNotFoundError(f"Dataset files not found in {save_path}")
    
    # Load npz
    data = np.load(npz_file)
    inputs = data['inputs']
    labels = data['labels']
    
    # Load metadata
    with open(meta_file, 'rb') as f:
        metadata = pickle.load(f)
    
    return inputs, labels, metadata

#======================================================================#
#