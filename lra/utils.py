#
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from pdebench.utils import make_optimizer_adamw, make_optimizer_lion, make_optimizer_muon

__all__ = [
    # OPTIMIZERS
    'make_optimizer_adamw',
    'make_optimizer_lion',
    'make_optimizer_muon',
    # COLLATORS
    'simple_collate',
    'PaddingCollate',
    # LOSSES
    'SequenceClassificationLoss',
    # STATS
    'ClassificationStatsFun',
]

#======================================================================#
# LOSSES
#======================================================================#
class SequenceClassificationLoss:
    def __init__(self, binary_classification: bool = False, ignore_index: int = -100):
        self.binary_classification = binary_classification
        self.ignore_index = ignore_index

    def __call__(self, trainer, model, batch):
        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch.get('attention_mask', None)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        return compute_cls_loss(logits=logits, labels=labels, binary_classification=self.binary_classification, ignore_index=self.ignore_index)

def compute_cls_loss(logits: torch.Tensor, labels: torch.Tensor, binary_classification: bool = False, ignore_index: int = -100):
    assert labels.dim() in [1, 2], f"Got labels.dim() = {labels.dim()} (size = {labels.size()}). Labels must be 1D or 2D for sequence classification or token classification."
    assert labels.size() == logits.size()[:-1], f"Got labels.size() = {labels.size()} and logits.size() = {logits.size()}. Logits shape must be [B, N, V] for tokenwise classification or [B, V] for token classification. For binary classification ({binary_classification}), V=1."
    
    # Binary SEQ: logits: [B, 1], labels: [B,]
    # Multi  SEQ: logits: [B, V], labels: [B,]
    # Binary TOK: logits: [B, N, 1], labels: [B, N]
    # Multi  TOK: logits: [B, N, V], labels: [B, N]

    if binary_classification:
        # logits: [B, N], or [B,]
        # labels: [B, N], or [B,]
        labels = labels.view(-1).float()
        logits = logits.view(-1)

        # Mask out ignored indices
        if ignore_index is not None:
            mask = (labels != ignore_index)
            labels = labels[mask]
            logits = logits[mask]
            if labels.numel() == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss = F.binary_cross_entropy_with_logits(logits, labels)
    else:
        # logits: [B, N, V], or [B, V]
        # labels: [B, N], or [B,]
        V = logits.size(-1)

        # flatten labels and logits
        # F.cross_entropy cannot handle per-token multi-label classification
        labels = labels.view(-1)
        logits = logits.view(-1, V)

        loss = F.cross_entropy(logits, labels, ignore_index=ignore_index)

    return loss

#======================================================================#
# SIMPLE COLLATE
#======================================================================#
@torch.no_grad()
def simple_collate(batch):
    batch = {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}
    return {**batch, 'attention_mask': None}

#======================================================================#
# PADDING COLLATE (for variable-length sequences)
#======================================================================#
class PaddingCollate:
    def __init__(self, pad_id: int, ignore_index: int):
        self.pad_id = pad_id
        self.ignore_index = ignore_index

    @torch.no_grad()
    def __call__(self, batch):
        # batch is list of dicts: {"input_ids": LongTensor[N_i], "labels": LongTensor[N_i]}
        input_seqs = [torch.as_tensor(x["input_ids"], dtype=torch.long) for x in batch]
        label_seqs = [torch.as_tensor(x["labels"], dtype=torch.long) for x in batch]

        input_ids = pad_sequence(input_seqs, batch_first=True, padding_value=self.pad_id)  # [B, N]

        if label_seqs[0].numel() == 1:
            labels = torch.cat(label_seqs, dim=0) # [B,]
        else:
            labels = pad_sequence(label_seqs, batch_first=True, padding_value=self.ignore_index)  # [B, N]

        attention_mask = (input_ids != self.pad_id)  # [B, N]

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

#======================================================================#
# STATS (classification accuracy)
#======================================================================#
class ClassificationStatsFun:
    def __init__(self, binary_classification: bool = False, ignore_index: int = None):
        self.binary_classification = binary_classification
        self.ignore_index = ignore_index

    def __call__(self, trainer, loader, split: str):
        model = trainer.model
        device = trainer.device

        if trainer.GLOBAL_RANK == 0:
            from tqdm import tqdm
            batch_iterator = tqdm(loader, desc="Evaluating (train/test) dataset", ncols=80)
        else:
            batch_iterator = loader

        loss_sum, n_items = 0.0, 0
        tok_correct, tok_total = 0, 0
        seq_correct, seq_total = 0, 0

        for batch in batch_iterator:
            batch = trainer.move_to_device(batch)
            with trainer.auto_cast:
                input_ids = batch['input_ids']
                attention_mask = batch.get('attention_mask', None)
                logits = model(input_ids=input_ids, attention_mask=attention_mask) # [B, V] or [B, N, V]
                labels = batch['labels']                                           # [B,] or [B, N]
                loss = compute_cls_loss(logits=logits, labels=labels, binary_classification=self.binary_classification, ignore_index=self.ignore_index)

            # Binary SEQ: logits: [B, 1], labels: [B,]
            # Multi  SEQ: logits: [B, V], labels: [B,]
            # Binary TOK: logits: [B, N, 1], labels: [B, N]
            # Multi  TOK: logits: [B, N, V], labels: [B, N]

            preds = (logits > 0.).squeeze(-1) if self.binary_classification else logits.argmax(dim=-1)
            assert labels.size() == preds.size(), f"Got labels.size() = {labels.size()} and preds.size() = {preds.size()}"

            if labels.dim() == 1:
                # labels is [B,]: per-sequence classification
                # Count non-ignored items for loss calculation
                valid_mask = (labels != self.ignore_index)
                n_items += valid_mask.sum().item()
                loss_sum += loss.item() * valid_mask.sum().item()
                seq_correct += (preds == labels).sum().item()
                seq_total += labels.size(0)
            else:
                # labels is [B, N]: per-token classification
                # Count non-ignored tokens for loss calculation
                valid_mask = (labels != self.ignore_index)
                n_valid = valid_mask.sum().item()
                n_items += n_valid
                loss_sum += loss.item() * n_valid

                # Only count predictions for non-ignored tokens
                valid_preds = preds[valid_mask]
                valid_labels = labels[valid_mask]
                tok_correct += (valid_preds == valid_labels).sum().item()
                tok_total += n_valid

                # For sequence accuracy, check if all non-ignored tokens match
                for b in range(labels.size(0)):
                    seq_valid_mask = (labels[b] != self.ignore_index)
                    if seq_valid_mask.any():
                        seq_preds = preds[b][seq_valid_mask]
                        seq_labels = labels[b][seq_valid_mask]
                        if (seq_preds == seq_labels).all():
                            seq_correct += 1
                    else:
                        # If all tokens are ignored, count as correct?
                        seq_correct += 1
                    seq_total += 1

        # Aggregate metrics across DDP processes
        if trainer.DDP:
            metrics = [
                ('loss_sum', loss_sum), ('n_items', n_items), 
                ('tok_correct', tok_correct), ('tok_total', tok_total),
                ('seq_correct', seq_correct), ('seq_total', seq_total),
            ]
            for name, val in metrics:
                t = torch.tensor(val, device=device, dtype=torch.float64)
                dist.all_reduce(t, dist.ReduceOp.SUM)
                if name == 'loss_sum':
                    loss_sum = t.item()
                elif name == 'n_items':
                    n_items = int(t.item())
                elif name == 'tok_correct':
                    tok_correct = int(t.item())
                elif name == 'tok_total':
                    tok_total = int(t.item())
                elif name == 'seq_correct':
                    seq_correct = int(t.item())
                elif name == 'seq_total':
                    seq_total = int(t.item())

        loss = loss_sum / max(1, n_items)
        seq_acc = seq_correct / max(1, seq_total)
        stats = dict(sequence_accuracy=seq_acc)
        if tok_total > 0:
            stats['token_accuracy'] = tok_correct / max(1, tok_total)

        return loss, stats

#======================================================================#
#