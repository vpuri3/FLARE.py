#
import os
import numpy as np
import torch
from torch.utils.data import Dataset

#======================================================================#
# Long Range Arena dataset
#======================================================================#
def build_lra_filename(task_name: str, split: str, data_root: str) -> str:
    if task_name == 'listops':
        base = 'lra-listops'
    elif task_name == 'text':
        base = 'lra-text'
    elif task_name == 'retrieval':
        base = 'lra-retrieval'
    elif task_name == 'image':
        base = 'lra-image'
    elif task_name == 'pathfinder32':
        base = 'lra-pathfinder32-curv_contour_length_14'
    elif task_name == 'pathfinder128':
        base = 'lra-pathfinder128-curv_contour_length_14'
    else:
        raise ValueError(f"Unknown LRA task: {task_name}")
    return os.path.join(data_root, f"{base}.{split}.pickle")

def scan_samples_for_metadata(samples, task: str):
    max_token = 0
    max_label = -1
    seq_len0 = None
    seq_len1 = None
    for ex in samples:

        assert 'label' in ex, f"Label not found in example: {ex}"
        assert 'input_ids_0' in ex, f"Input IDs not found in example: {ex}"

        max_label = max(max_label, int(ex['label']))

        if task == 'retrieval':
            a = ex['input_ids_0']
            b = ex['input_ids_1']

            if seq_len0 is None:
                seq_len0 = int(len(a))
                seq_len1 = int(len(b))
            else:
                assert seq_len0 == int(len(a)), f"Sequence length must be equal for retrieval. Got {seq_len0} and {int(len(a))}."
                assert seq_len1 == int(len(b)), f"Sequence length must be equal for retrieval. Got {seq_len1} and {int(len(b))}."

            assert seq_len0 == seq_len1, f"Sequence lengths must be equal for retrieval. Got {seq_len0} and {seq_len1}."

            if isinstance(a, np.ndarray):
                max_token = max(max_token, int(a.max(initial=0)), int(b.max(initial=0)))
            else:
                max_token = max(max_token, max(a), max(b))

        else:
            a = ex['input_ids_0']
            if seq_len0 is None:
                seq_len0 = int(len(a))
            else:
                assert seq_len0 == int(len(a)), f"Sequence length must be equal for retrieval. Got {seq_len0} and {int(len(a))}."

            if isinstance(a, np.ndarray):
                max_token = max(max_token, int(a.max(initial=0)))
            else:
                max_token = max(max_token, max(a))

    return max_token, max_label, seq_len0

class LRADataset(Dataset):
    def __init__(self, items, task: str, seq_len: int):
        self.items = items
        self.is_pair = task == 'retrieval'
        self.seq_len = seq_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ex = self.items[idx]
        if self.is_pair:
            a = torch.as_tensor(ex['input_ids_0'], dtype=torch.long)
            b = torch.as_tensor(ex['input_ids_1'], dtype=torch.long)
            input_ids = torch.cat([a, b], dim=0)
            out = dict(input_ids=input_ids)
        else:
            a = torch.as_tensor(ex['input_ids_0'], dtype=torch.long)
            out = dict(input_ids=a)
        if 'label' in ex:
            out['labels'] = torch.as_tensor(int(ex['label']), dtype=torch.long)
        return out
#======================================================================#
#