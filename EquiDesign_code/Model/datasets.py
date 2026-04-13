import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from tqdm import tqdm

# Mapping of amino acid to single letter code and lookup indices
abbrev = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E", "GLN": "Q", "GLY": "G", 
          "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", 
          "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
lookup = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9, 'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8, 
          'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 'N': 2, 'Y': 18, 'M': 12, '-': 20, 'X': 20}

class DynamicLoader(Dataset):
    def __init__(self, dataset, batch_size=3000, shuffle=True):
        """
        Initializes DynamicLoader.
        Args:
            dataset: A list of data points (dicts).
            batch_size: Maximum batch size (number of amino acids).
            shuffle: Whether to shuffle the dataset before batching.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.clusters = []
        self.batch()

    def batch(self):
        """ Create dynamic batches based on sequence lengths """
        lengths = [len(b['seq']) for b in self.dataset]
        clusters, batch = [], []

        # Sort by sequence length to optimize batching
        for ix in np.argsort(lengths):
            size = lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
            else:
                if len(batch) > 0:
                    clusters.append(batch)
                batch = [ix]
        if len(batch) > 0:
            clusters.append(batch)

        self.clusters = clusters
        print(len(clusters), 'batches created from', len(self.dataset), 'structures.')

    def __len__(self):
        """ Return the number of batches """
        return len(self.clusters)

    def __getitem__(self, idx):
        """ Get a batch by its index """
        batch_idx = self.clusters[idx]
        batch = [self.dataset[i] for i in batch_idx]
        return self.parse_batch(batch)

    def parse_batch(self, batch):
        """ Convert a batch to PyTorch tensors """
        B = len(batch)
        L_max = max([len(b['seq']) for b in batch])  # Max sequence length in the batch
        X = np.zeros([B, L_max, 4, 3], dtype=np.float32)  # Coordinates (4 atoms: N, CA, C, O)
        S = np.zeros([B, L_max], dtype=np.int32)  # Sequence
        # SS: H=0, E=1, C=2 -> one-hot [B, L_max, 3]
        SS = np.zeros([B, L_max, 3], dtype=np.float32)

        ss_map = {'H': 0, 'E': 1, 'C': 2}

        for i, b in enumerate(batch):
            l = len(b['seq'])
            x = b['coords']

            if isinstance(x, dict):  # Convert dict to numpy array
                x = np.stack([x[c] for c in ['N', 'CA', 'C', 'O']], axis=1)

            X[i] = np.pad(x, [[0, L_max - l], [0, 0], [0, 0]], 'constant', constant_values=(np.nan,))
            S[i, :l] = np.asarray([lookup.get(a, 20) for a in b['seq']], dtype=np.int32)

            ss_str = b.get('ss', 'C' * l)
            for j, c in enumerate(ss_str[:l]):
                k = ss_map.get(c, 2)
                SS[i, j, k] = 1.0

        isnan = np.isnan(X)
        mask = np.isfinite(np.sum(X, axis=(2, 3))).astype(np.float32)
        X[isnan] = 0.0
        X = np.nan_to_num(X)

        return (torch.tensor(X, dtype=torch.float32), torch.tensor(S, dtype=torch.int32),
                torch.tensor(mask, dtype=torch.float32), torch.tensor(SS, dtype=torch.float32))

    def get_data_loader(self, batch_size=32, shuffle=True):
        """ Return a DataLoader for this dataset """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        """ PyTorch collate function """
        return self.parse_batch(batch)


def load_dataset(path, batch_size, shuffle=True):
    """ Load dataset from a JSON file """
    with open(path, 'r') as f:
        data = json.load(f)
    data = DynamicLoader(data, batch_size, shuffle)
    return data

def ts50_dataset(batch_size):
    """ Load TS50 dataset """
    return load_dataset('../data/ts50.json', batch_size)

def load_dataset1(path, batch_size, shuffle=True):
    """ Load dataset from a JSONL file """
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]  # 每行一个 JSON
    data = DynamicLoader(data, batch_size, shuffle)
    return data

def single_sample(jsonl_path, batch_size=1):
    """ Load single sample from a JSONL file """
    return load_dataset1(jsonl_path, batch_size, shuffle=False)

def ts500_dataset(batch_size):
    "Load TS500 dataset"
    return load_dataset('../data/ts500.json', batch_size)

def cath_dataset(batch_size, jsonl_file='../data/chain_set.jsonl', split_file='../data/chain_set_splits.json',
                 ss_file='../data/chain_set_ss.json', filter_file=None):
    """ Load and split CATH dataset into train, validation, and test sets """
    with open(split_file, 'r') as f:
        dataset_splits = json.load(f)

    if filter_file:
        with open(filter_file, 'r') as f:
            filter_data = json.load(f)
        filter_file = filter_data.get("test", [])

    # 加载二级结构（若有）
    ss_dict = {}
    ss_path = os.path.normpath(os.path.join(os.path.dirname(jsonl_file), os.path.basename(ss_file)))
    if os.path.exists(ss_path):
        with open(ss_path, 'r') as f:
            ss_dict = json.load(f)
        print(f"Loaded secondary structure for {len(ss_dict)} chains from {ss_path}")
    else:
        print(f"Note: {ss_path} not found, SS will default to coil. Run precompute_chain_set_ss.py first.")

    # Initialize train, validation, test sets
    train_list, val_list, test_list = dataset_splits['train'], dataset_splits['validation'], dataset_splits['test']
    trainset, valset, testset = [], [], []

    with open(jsonl_file, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        entry = json.loads(line)
        seq = entry['seq']
        name = entry['name']
        if filter_file and name not in filter_file:
            continue
        for key, val in entry['coords'].items():
            entry['coords'][key] = np.asarray(val)
        # 添加二级结构，若无则用全 C（coil）
        entry['ss'] = ss_dict.get(name, 'C' * len(seq))
        if name in train_list: trainset.append(entry)
        elif name in val_list: valset.append(entry)
        elif name in test_list: testset.append(entry)

    trainset = DynamicLoader(trainset, batch_size)
    valset = DynamicLoader(valset, batch_size)
    testset = DynamicLoader(testset, batch_size)

    return trainset, valset, testset
