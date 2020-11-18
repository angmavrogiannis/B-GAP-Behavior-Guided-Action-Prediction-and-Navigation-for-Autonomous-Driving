import os.path as osp

import torch
from scipy.io import loadmat
from torch_geometric.data import Data, InMemoryDataset, download_url


class SuiteSparseMatrixCollection(InMemoryDataset):
    r"""A suite of sparse matrix benchmarks known as the `Suite Sparse Matrix
    Collection <https://sparse.tamu.edu>`_ collected from a wide range of
    applications.

    Args:
        root (string): Root directory where the dataset should be saved.
        group (string): The group of the sparse matrix.
        name (string): The name of the sparse matrix.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://sparse.tamu.edu/mat/{}/{}.mat'

    def __init__(self, root, group, name, transform=None, pre_transform=None):
        self.group = group
        self.name = name
        super(SuiteSparseMatrixCollection,
              self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.group, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.group, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.name}.mat'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        url = self.url.format(self.group, self.name)
        download_url(url, self.raw_dir)

    def process(self):
        mat = loadmat(self.raw_paths[0])['Problem'][0][0][2].tocsr().tocoo()

        row = torch.from_numpy(mat.row).to(torch.long)
        col = torch.from_numpy(mat.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = torch.from_numpy(mat.data).to(torch.float)
        if torch.all(edge_attr == 1.):
            edge_attr = None

        size = torch.Size(mat.shape)
        if mat.shape[0] == mat.shape[1]:
            size = None

        num_nodes = mat.shape[0]

        data = Data(edge_index=edge_index, edge_attr=edge_attr, size=size,
                    num_nodes=num_nodes)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}(group={}, name={})'.format(self.__class__.__name__,
                                              self.group, self.name)
