import os
import datetime

import torch

from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_gz)


class BitcoinOTC(InMemoryDataset):
    r"""The Bitcoin-OTC dataset from the `"EvolveGCN: Evolving Graph
    Convolutional Networks for Dynamic Graphs"
    <https://arxiv.org/abs/1902.10191>`_ paper, consisting of 138
    who-trusts-whom networks of sequential time steps.

    Args:
        root (string): Root directory where the dataset should be saved.
        edge_window_size (int, optional): The window size for the existence of
            an edge in the graph sequence since its initial creation.
            (default: :obj:`10`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz'

    def __init__(self,
                 root,
                 edge_window_size=10,
                 transform=None,
                 pre_transform=None):
        self.edge_window_size = edge_window_size
        super(BitcoinOTC, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'soc-sign-bitcoinotc.csv'

    @property
    def processed_file_names(self):
        return 'data.pt'

    @property
    def num_nodes(self):
        return self.data.edge_index.max().item() + 1

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_gz(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[:-1]
            data = [[x for x in line.split(',')] for line in data]

            edge_index = [[int(line[0]), int(line[1])] for line in data]
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_index = edge_index - edge_index.min()
            edge_index = edge_index.t().contiguous()
            num_nodes = edge_index.max().item() + 1

            edge_attr = [float(line[2]) for line in data]
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)

            stamps = [int(float(line[3])) for line in data]
            stamps = [datetime.datetime.fromtimestamp(x) for x in stamps]

        offset = datetime.timedelta(days=13.8)  # Results in 138 time steps.
        graph_idx, factor = [], 1
        for t in stamps:
            factor = factor if t < stamps[0] + factor * offset else factor + 1
            graph_idx.append(factor - 1)
        graph_idx = torch.tensor(graph_idx, dtype=torch.long)

        data_list = []
        for i in range(graph_idx.max().item() + 1):
            mask = (graph_idx > (i - self.edge_window_size)) & (graph_idx <= i)
            data = Data()
            data.edge_index = edge_index[:, mask]
            data.edge_attr = edge_attr[mask]
            data.num_nodes = num_nodes
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
