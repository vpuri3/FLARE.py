#
import os
import random
import itertools
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import Dataset

#======================================================================#
# Strassen attention tasks
# https://arxiv.org/pdf/2501.19215v2
# https://anonymous.4open.science/r/strassen-attention-neurips25-434F/README.md
#======================================================================#
class FunctCompDataset(Dataset):
    def __init__(
        self,
        DATA_ROOT: str = None,
        GLOBAL_RANK: int = 0,
        num_instances: int = 50_000,
        min_seq_len: int = 25,
        max_seq_len: int = 30,
        seed: int = 42,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.data, self.labels = self.load_or_generate_dataset(DATA_ROOT, GLOBAL_RANK, num_instances, min_seq_len, max_seq_len)

    @staticmethod
    def load_or_generate_dataset(DATA_ROOT: str, GLOBAL_RANK: int, num_instances: int, min_seq_len: int, max_seq_len: int):
        """Load dataset from DATA_ROOT if exists, else generate it (only if GLOBAL_RANK == 0)."""
        if DATA_ROOT is not None:
            os.makedirs(DATA_ROOT, exist_ok=True)
            pkl_file = Path(DATA_ROOT) / 'function_composition.pkl'
            
            if pkl_file.exists():
                print(f"Loading cached function_composition dataset from {DATA_ROOT}")
                with open(pkl_file, 'rb') as f:
                    data, labels = pickle.load(f)
                return data, labels
            
            # If not rank 0, wait for rank 0 to generate and save the dataset
            if GLOBAL_RANK != 0:
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                # After barrier, file should exist (rank 0 will have saved it)
                if pkl_file.exists():
                    print(f"Rank {GLOBAL_RANK}: Loading dataset after generation...")
                    with open(pkl_file, 'rb') as f:
                        data, labels = pickle.load(f)
                    return data, labels
                else:
                    raise RuntimeError(f"Rank {GLOBAL_RANK}: Dataset file not found after barrier")

        # Only generate if rank 0
        if GLOBAL_RANK != 0:
            raise RuntimeError(f"Rank {GLOBAL_RANK}: Dataset not found and generation should be done by rank 0")
        
        print(f"Generating function_composition dataset")
        data = []
        labels = []

        for i in tqdm(range(num_instances), desc="Generating sequences", ncols=80):
            label = random.choice([0,1])
            seq_len_obj = random.choice(range(min_seq_len, max_seq_len + 1))

            seq = np.random.randint(low=0,high=seq_len_obj,size=seq_len_obj,dtype=int)
                
            if label:
                if seq[seq[0]] != 0:
                    seq[seq[0]] = 0
            else:
                if seq[seq[0]] == 0:
                    index_pos = [i for i in np.arange(1,seq_len_obj) if seq[i] != 0 ]
                    if len(index_pos) == 0:
                        pos_no_zero = random.choice(np.arange(1,seq_len_obj))
                        seq[pos_no_zero] = random.choice(np.arange(1,seq_len_obj))
                    else:
                        rnd_numb = random.choice(index_pos)
                    seq[seq[0]] = rnd_numb

            # Save as list to preserve variable length (no padding)
            data.append(seq.tolist())
            labels.append([label])
        
        # Save if DATA_ROOT provided, then barrier to sync before other ranks load
        if DATA_ROOT is not None:
            with open(pkl_file, 'wb') as f:
                pickle.dump((data, labels), f)
            # Synchronize after saving: rank 0 waits here, others proceed to load
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        
        return data, labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        label = self.labels[idx]
        
        return dict(
            input_ids=torch.tensor(seq, dtype=torch.long).view(-1),
            labels=torch.tensor(label, dtype=torch.long).view(-1),
        )

class BinaryRelationCompDataset(Dataset):
    def __init__(
        self,
        DATA_ROOT: str = None,
        GLOBAL_RANK: int = 0,
        num_graphs: int = 50_000,
        minimun_nodes: int = 6,
        maximun_nodes: int = 8,
        p_edge: float = 0.325,
        seed: int = 42,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.n_min = minimun_nodes
        self.n_max = maximun_nodes
        self.p_edge = p_edge
        
        self.data, self.labels = self.load_or_generate_dataset(DATA_ROOT, GLOBAL_RANK, num_graphs)
    
    def generate_random_graph(self, n, p):
        adjacency_matrix = np.random.binomial(1, p, size=(n, n))
        return adjacency_matrix
    
    def is_related(self, adj_matrix, i, j):
        result = 0
        neighbors_i = np.where(adj_matrix[i] == 1)[0]       # Neighbors of i
        neighbors_j = np.where(adj_matrix[:, j] == 1)[0]    # Neighbors of j

        for u in neighbors_i:
            for v in neighbors_j:
                if u == v:
                    result = 1
                    break
            if result:
                break
        return result
    
    def load_or_generate_dataset(self, DATA_ROOT: str, GLOBAL_RANK: int, num_graphs: int):
        """Load dataset from DATA_ROOT if exists, else generate it (only if GLOBAL_RANK == 0)."""
        if DATA_ROOT is not None:
            os.makedirs(DATA_ROOT, exist_ok=True)
            pkl_file = Path(DATA_ROOT) / 'binary_relation_composition.pkl'
            
            if pkl_file.exists():
                print(f"Loading cached binary_relation_composition dataset from {DATA_ROOT}")
                with open(pkl_file, 'rb') as f:
                    data, labels = pickle.load(f)
                return data, labels
            
            # If not rank 0, wait for rank 0 to generate and save the dataset
            if GLOBAL_RANK != 0:
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                # After barrier, file should exist (rank 0 will have saved it)
                if pkl_file.exists():
                    print(f"Rank {GLOBAL_RANK}: Loading dataset after generation...")
                    with open(pkl_file, 'rb') as f:
                        data, labels = pickle.load(f)
                    return data, labels
                else:
                    raise RuntimeError(f"Rank {GLOBAL_RANK}: Dataset file not found after barrier")
        
        # Only generate if rank 0
        if GLOBAL_RANK != 0:
            raise RuntimeError(f"Rank {GLOBAL_RANK}: Dataset not found and generation should be done by rank 0")
        
        print(f"Generating binary_relation_composition dataset")
        data = []
        labels = []
        total_ones = 0
        total_instances = 0
        
        for _ in tqdm(range(num_graphs), desc="Generating graphs", ncols=80):
            number_nodes = random.randint(self.n_min, self.n_max)
            adjacency_matrix = self.generate_random_graph(number_nodes, self.p_edge)
            # Save as list (will be flattened in __getitem__)
            data.append(adjacency_matrix.tolist())
            
            targets = [
                self.is_related(adjacency_matrix, i, j) 
                for i in range(number_nodes)
                for j in range(number_nodes)
            ]

            total_ones += np.sum([tar for tar in targets if tar == 1])
            total_instances += len([tar for tar in targets if tar > -100])
            # Save labels as list of lists (variable length per graph)
            labels.append(np.array(targets).reshape(number_nodes, number_nodes).tolist())

        print("Fraction of 1s: ", total_ones / total_instances)

        # Save if DATA_ROOT provided, then barrier to sync before other ranks load
        if DATA_ROOT is not None:
            with open(pkl_file, 'wb') as f:
                pickle.dump((data, labels), f)
            # Synchronize after saving: rank 0 waits here, others proceed to load
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        am = self.data[idx]  # adjacency matrix saved as list of lists [[...], [...]]
        label = self.labels[idx]  # label matrix saved as list of lists

        # Convert to numpy array for flattening, then to list to preserve variable length
        am_array = np.array(am)
        label_array = np.array(label)

        # Flatten adjacency matrix to input_ids
        input_ids = am_array.flatten()
        # Flatten labels
        labels_flat = label_array.flatten()

        return dict(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            labels=torch.tensor(labels_flat, dtype=torch.long)
        )

class Match3Dataset(Dataset):
    def __init__(
        self,
        DATA_ROOT: str = None,
        GLOBAL_RANK: int = 0,
        num_instances: int = 50_000,
        min_seq_len: int = 30,
        max_seq_len: int = 35,
        M: int = 37,
        num_bins: int = 4,
        seed: int = 42,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.M = M
        self.num_bins = num_bins
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        
        self.data, self.labels = self.load_or_generate_dataset(DATA_ROOT, GLOBAL_RANK, num_instances)
    
    def load_or_generate_dataset(self, DATA_ROOT: str, GLOBAL_RANK: int, num_instances: int):
        """Load dataset from DATA_ROOT if exists, else generate it (only if GLOBAL_RANK == 0)."""
        if DATA_ROOT is not None:
            os.makedirs(DATA_ROOT, exist_ok=True)
            pkl_file = Path(DATA_ROOT) / 'match3.pkl'
            
            if pkl_file.exists():
                print(f"Loading cached match3 dataset from {DATA_ROOT}")
                with open(pkl_file, 'rb') as f:
                    data, labels = pickle.load(f)
                return data, labels
            
            # If not rank 0, wait for rank 0 to generate and save the dataset
            if GLOBAL_RANK != 0:
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                # After barrier, file should exist (rank 0 will have saved it)
                if pkl_file.exists():
                    print(f"Rank {GLOBAL_RANK}: Loading dataset after generation...")
                    with open(pkl_file, 'rb') as f:
                        data, labels = pickle.load(f)
                    return data, labels
                else:
                    raise RuntimeError(f"Rank {GLOBAL_RANK}: Dataset file not found after barrier")
        
        # Only generate if rank 0
        if GLOBAL_RANK != 0:
            raise RuntimeError(f"Rank {GLOBAL_RANK}: Dataset not found and generation should be done by rank 0")
        
        print(f"Generating match3 dataset")
        data = []
        labels = []
        
        possible_values = np.arange(1, self.M)

        # Dynamically generate bins based on num_bins parameter
        containers = {f"{i * (100 // self.num_bins)}-{(i + 1) * (100 // self.num_bins)}%": [] for i in range(self.num_bins)}

        # Target number of sequences per subset (divide by num_bins)
        target_per_subset = num_instances // self.num_bins
        remainder = num_instances % self.num_bins

        # Create a map to adjust target sequence count per bin
        target_per_subset_map = {key: target_per_subset for key in containers}
        # Distribute the remainder across the bins
        for i, key in enumerate(containers):
            if i < remainder:
                target_per_subset_map[key] += 1

        total_ones, total_numbers = 0, 0
        
        rnd_instances = 5000

        if num_instances < rnd_instances:
            iter = num_instances
        else:
            iter = rnd_instances

        MAX_PERC_ONES = 40

        for i in tqdm(range(iter), desc="Generating initial sequences", ncols=80):
            seq_len_obj = random.choice(range(self.min_seq_len, self.max_seq_len + 1))
            perc_ones = random.choice(range(1, MAX_PERC_ONES)) 
            num_ones = int(seq_len_obj * perc_ones / 100)

            seq = self.generate_sequence(seq_len_obj,
                                            values=possible_values,
                                            num_ones=num_ones)  # Generate sequence
            targets, sum_ones = self.get_targets(seq) 

            percent_ones = int(100 * sum_ones / seq_len_obj)

            # Determine the bin for the sequence based on percent_ones
            bin_index = int(percent_ones // (100 / self.num_bins))
            
            if bin_index == self.num_bins:  # Ensure we don't exceed the last bin
                bin_index -= 1
            container_key = f"{bin_index * (100 // self.num_bins)}-{(bin_index + 1) * (100 // self.num_bins)}%"
            
            if len(containers[container_key]) < target_per_subset_map[container_key]:
                containers[container_key].append((seq, targets))
                total_ones += sum_ones
                total_numbers += seq_len_obj

        for key, container in tqdm(containers.items(), desc="Augmenting bins", ncols=80):
            # Check if the bin still needs more sequences
            while len(container) < target_per_subset_map[key]:
                # Sample sequences and their labels from the existing bin
                sampled_seq, sampled_targets = random.choice(container)

                # Generate a permutation index
                permutation_index = np.random.permutation(len(sampled_seq))

                # Permute both the sequence and the labels using the same index
                permuted_seq = sampled_seq[permutation_index]
                permuted_targets = sampled_targets[permutation_index]

                # Update totals for ones and total numbers
                total_ones += np.sum(permuted_targets)
                total_numbers += len(permuted_seq)

                # Add the permuted sequence and targets to the bin
                containers[key].append((permuted_seq, permuted_targets))

            # Check if all containers have enough samples
            if all(len(containers[key]) >= target_per_subset_map[key] for key in containers):
                break
        
        # Now balance the dataset by sampling equally from each container
        for _, sequences in tqdm(containers.items(), desc="Balancing dataset", ncols=80):
            for seq, target in sequences:
                # Save as lists to preserve variable length (no padding)
                data.append(seq.tolist() if isinstance(seq, np.ndarray) else list(seq))
                labels.append(target.tolist() if isinstance(target, np.ndarray) else list(target))
        
        # Save if DATA_ROOT provided, then barrier to sync before other ranks load
        if DATA_ROOT is not None:
            with open(pkl_file, 'wb') as f:
                pickle.dump((data, labels), f)
            # Synchronize after saving: rank 0 waits here, others proceed to load
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        
        return data, labels

    def update_candidates(self,seq,values):
        size = len(seq)
        tup = list(itertools.combinations(seq, 2)) + [(seq[i],seq[i]) for i in range(size)]
        sum_jk = np.unique([np.sum(k) for k in tup ])

        label1_cand = [k for k in values for xjxk in sum_jk if (k + xjxk)%self.M == 0]
        label1_cand = np.unique(label1_cand + [k for k in values if 3*k % self.M == 0 ])
        label0_cand = np.setdiff1d( values,label1_cand )       

        return label1_cand,label0_cand

    def generate_sequence(self, size, values, num_ones):
        seq = np.array([])
        idx_ones = random.choices(range(2, size), k = num_ones)
        labels = []

        label1_cand = []
        label0_cand = np.array(values)
        ones_cnt = 0
        
        for i in range(size):
            if i > 0:
                if i in idx_ones and len(label1_cand) > 0:
                    seq = np.concatenate( ( seq,np.array([ random.choice( label1_cand )  ]) ) )
                    labels.append(1)
                elif len(label0_cand) > 0:
                    seq = np.concatenate( ( seq,np.array([ random.choice(label0_cand) ]) ) )
                    labels.append(0)
                else:
                    seq = np.concatenate( ( seq,np.array([ random.choice(label1_cand) ]) ) )
                
                label1_cand,label0_cand = self.update_candidates(seq,values)

            else:
                seq = np.concatenate( (seq,[random.choice(values)]) )
                label1_cand,label0_cand = self.update_candidates(seq,values)
                labels.append(0)
            
            _, ones_cnt = self.get_targets(seq=seq)
            if ones_cnt >= num_ones:
                while len(seq) < size:
                    if len(label0_cand) > 0:
                        seq = np.concatenate( ( seq,np.array([ random.choice(label0_cand) ]) ) )
                        label1_cand,label0_cand = self.update_candidates(seq,values)
                    else:
                        seq = np.concatenate( ( seq,np.array([ random.choice(label1_cand) ]) ) )
                break
                
        permutation_index = np.random.permutation(size)
        seq = seq[permutation_index]
        return seq

    def get_targets(self, seq):
        targets = np.zeros_like(seq)

        for i in range(len(seq)):
            for j in range(len(seq)):
                for k in range(len(seq)):
                    if (seq[i] + seq[j] + seq[k]) % self.M == 0:
                        targets[i] = 1
                        break
        return targets, np.sum(targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        labels = self.labels[idx]
        
        return dict(
            input_ids=torch.tensor(seq, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long)
        )

class QuotientBinaryRelationCompDataset(Dataset):
    def __init__(
        self,
        DATA_ROOT: str = None,
        GLOBAL_RANK: int = 0,
        num_graphs: int = 50_000,
        minimun_nodes: int = 6,
        maximun_nodes: int = 8,
        n_colors: int = 9,
        p_edge: float = 0.433,
        seed: int = 42,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.n_max = maximun_nodes
        self.n_min = minimun_nodes
        self.M = n_colors
        self.p_edge = p_edge

        self.data, self.labels = self.load_or_generate_dataset(DATA_ROOT, GLOBAL_RANK, num_graphs)
    
    def load_or_generate_dataset(self, DATA_ROOT: str, GLOBAL_RANK: int, num_graphs: int):
        """Load dataset from DATA_ROOT if exists, else generate it (only if GLOBAL_RANK == 0)."""
        if DATA_ROOT is not None:
            os.makedirs(DATA_ROOT, exist_ok=True)
            pkl_file = Path(DATA_ROOT) / 'quotient_binary_relation_composition.pkl'
            
            if pkl_file.exists():
                print(f"Loading cached quotient_binary_relation_composition dataset from {DATA_ROOT}")
                with open(pkl_file, 'rb') as f:
                    data, labels = pickle.load(f)
                return data, labels
            
            # If not rank 0, wait for rank 0 to generate and save the dataset
            if GLOBAL_RANK != 0:
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                # After barrier, file should exist (rank 0 will have saved it)
                if pkl_file.exists():
                    print(f"Rank {GLOBAL_RANK}: Loading dataset after generation...")
                    with open(pkl_file, 'rb') as f:
                        data, labels = pickle.load(f)
                    return data, labels
                else:
                    raise RuntimeError(f"Rank {GLOBAL_RANK}: Dataset file not found after barrier")
        
        # Only generate if rank 0
        if GLOBAL_RANK != 0:
            raise RuntimeError(f"Rank {GLOBAL_RANK}: Dataset not found and generation should be done by rank 0")
        
        print(f"Generating quotient_binary_relation_composition dataset")
        data = []
        labels = []
        total_eccns = 0
        total_instances = 0

        for _ in tqdm(range(num_graphs), desc="Generating graphs", ncols=80):
            number_nodes = random.randint(self.n_min, self.n_max)

            adjacency_matrix, node_colors = self.generate_random_graph(
                number_nodes,
                self.p_edge,
                number_nodes, # TODO: self.M (orig) number_nodes (paper)
            )
            
            # Save as list (will be flattened in __getitem__)
            data.append(adjacency_matrix.tolist())
            
            targets = [
                self.is_eccn(adjacency_matrix, node_colors, i, j) if i != j else -100
                for i in range(number_nodes)
                for j in range(number_nodes)
            ]

            total_eccns += np.sum([tar for tar in targets if tar == 1])
            total_instances += len([tar for tar in targets if tar > -100])
            # Save labels as list of lists (variable length per graph)
            labels.append(np.array(targets).reshape(number_nodes, number_nodes).tolist())
        
        print("ECCN fraction of 1s: ", total_eccns / total_instances)
        
        # Save if DATA_ROOT provided, then barrier to sync before other ranks load
        if DATA_ROOT is not None:
            with open(pkl_file, 'wb') as f:
                pickle.dump((data, labels), f)
            # Synchronize after saving: rank 0 waits here, others proceed to load
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        
        return data, labels

    def generate_random_graph(self, n, p, num_colors):
        adjacency_matrix = np.random.binomial(1, p, size=(n, n))
        np.fill_diagonal(adjacency_matrix, 0)
        node_colors = np.random.randint(1, num_colors + 1, size=n)
        return adjacency_matrix, node_colors


    def is_eccn(self, adj_matrix, node_colors, i, j):
        n = adj_matrix.shape[0]
        result = 0

        neighbors_i = np.where(adj_matrix[i] == 1)[0]  # Neighbors of i
        neighbors_j = np.where(adj_matrix[j] == 1)[0]  # Neighbors of j

        for u in neighbors_i:
            for v in neighbors_j:
                if node_colors[u] == node_colors[v] and u != v:
                    result = 1
                    break
            if result:
                break

        return result

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        am = self.data[idx]  # adjacency matrix saved as list of lists [[...], [...]]
        label = self.labels[idx]  # label matrix saved as list of lists
        
        # Convert to numpy array for flattening, then to list to preserve variable length
        am_array = np.array(am)
        label_array = np.array(label)
        
        # Flatten adjacency matrix to input_ids
        input_ids = am_array.flatten()
        # Flatten labels
        labels_flat = label_array.flatten()
        
        return dict(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            labels=torch.tensor(labels_flat, dtype=torch.long)
        )

#======================================================================#
#