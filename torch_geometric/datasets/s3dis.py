import os
import os.path as osp
import shutil

import h5py
import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)


class S3DIS(InMemoryDataset):
    r"""The (pre-processed) Stanford Large-Scale 3D Indoor Spaces dataset from
    the `"3D Semantic Parsing of Large-Scale Indoor Spaces"
    <http://buildingparser.stanford.edu/images/3D_Semantic_Parsing.pdf>`_
    paper, containing point clouds of six large-scale indoor parts in three
    buildings with 12 semantic elements (and one clutter class).

    Args:
        root (string): Root directory where the dataset should be saved.
        test_area (int, optional): Which area to use for testing (1-6).
            (default: :obj:`6`)
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = ('https://shapenet.cs.stanford.edu/media/'
           'indoor3d_sem_seg_hdf5_data.zip')

    def __init__(self,
                 root,
                 test_area=6,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        assert test_area >= 1 and test_area <= 6
        self.test_area = test_area
        super(S3DIS, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['all_files.txt', 'room_filelist.txt']

    @property
    def processed_file_names(self):
        test_area = self.test_area
        return ['{}_{}.pt'.format(s, test_area) for s in ['train', 'test']]

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        name = self.url.split(os.sep)[-1].split('.')[0]
        os.rename(osp.join(self.root, name), self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            filenames = [x.split('/')[-1] for x in f.read().split('\n')[:-1]]

        with open(self.raw_paths[1], 'r') as f:
            rooms = f.read().split('\n')[:-1]

        xs, ys = [], []
        for filename in filenames:
            f = h5py.File(osp.join(self.raw_dir, filename))
            xs += torch.from_numpy(f['data'][:]).unbind(0)
            ys += torch.from_numpy(f['label'][:]).to(torch.long).unbind(0)

        test_area = 'Area_{}'.format(self.test_area)
        train_data_list, test_data_list = [], []
        for i, (x, y) in enumerate(zip(xs, ys)):
            data = Data(pos=x[:, :3], x=x[:, 3:], y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            if test_area not in rooms[i]:
                train_data_list.append(data)
            else:
                test_data_list.append(data)

        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(test_data_list), self.processed_paths[1])
