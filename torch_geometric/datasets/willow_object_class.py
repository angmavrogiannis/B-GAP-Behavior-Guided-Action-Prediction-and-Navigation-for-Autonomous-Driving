import os
import os.path as osp
import shutil
import glob

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.io import loadmat
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)

try:
    import torchvision.models as models
    import torchvision.transforms as T
    from PIL import Image
except ImportError:
    models = None
    T = None
    Image = None


class WILLOWObjectClass(InMemoryDataset):
    r"""The WILLOW-ObjectClass dataset from the `"Learning Graphs to Match"
    <https://www.di.ens.fr/willow/pdfscurrent/cho2013.pdf>`_ paper,
    containing 10 equal keypoints of at least 40 images in each category.
    The keypoints contain interpolated features from a pre-trained VGG16 model
    on ImageNet (:obj:`relu4_2` and :obj:`relu5_1`).

    Args:
        root (string): Root directory where the dataset should be saved.
        category (string): The category of the images (one of :obj:`"Car"`,
            :obj:`"Duck"`, :obj:`"Face"`, :obj:`"Motorbike"`,
            :obj:`"Winebottle"`).
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
    url = ('http://www.di.ens.fr/willow/research/graphlearning/'
           'WILLOW-ObjectClass_dataset.zip')

    categories = ['face', 'motorbike', 'car', 'duck', 'winebottle']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32

    def __init__(self, root, category, transform=None, pre_transform=None,
                 pre_filter=None):
        assert category.lower() in self.categories
        self.category = category
        super(WILLOWObjectClass, self).__init__(root, transform, pre_transform,
                                                pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.category.capitalize(), 'processed')

    @property
    def raw_file_names(self):
        return [category.capitalize() for category in self.categories]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        os.unlink(osp.join(self.root, 'README'))
        os.unlink(osp.join(self.root, 'demo_showAnno.m'))
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(self.root, 'WILLOW-ObjectClass'), self.raw_dir)

    def process(self):
        if models is None or T is None or Image is None:
            raise ImportError('Package `torchvision` could not be found.')

        category = self.category.capitalize()
        names = glob.glob(osp.join(self.raw_dir, category, '*.png'))
        names = sorted([name[:-4] for name in names])

        vgg16_outputs = []

        def hook(module, x, y):
            vgg16_outputs.append(y.to('cpu'))

        vgg16 = models.vgg16(pretrained=True).to(self.device)
        vgg16.eval()
        vgg16.features[20].register_forward_hook(hook)  # relu4_2
        vgg16.features[25].register_forward_hook(hook)  # relu5_1

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        data_list = []
        for name in names:
            pos = loadmat('{}.mat'.format(name))['pts_coord']
            x, y = torch.from_numpy(pos).to(torch.float)
            pos = torch.stack([x, y], dim=1)

            # The "face" category contains a single image with less than 10
            # keypoints, so we need to skip it.
            if pos.size(0) != 10:
                continue

            with open('{}.png'.format(name), 'rb') as f:
                img = Image.open(f).convert('RGB')

            # Rescale keypoints.
            pos[:, 0] = pos[:, 0] * 256.0 / (img.size[0])
            pos[:, 1] = pos[:, 1] * 256.0 / (img.size[1])

            img = img.resize((256, 256), resample=Image.BICUBIC)
            img = transform(img)

            data = Data(img=img, pos=pos, name=name)
            data_list.append(data)

        imgs = [data.img for data in data_list]
        loader = DataLoader(imgs, self.batch_size, shuffle=False)
        for i, batch_img in enumerate(loader):
            vgg16_outputs.clear()

            with torch.no_grad():
                vgg16(batch_img.to(self.device))

            out1 = F.interpolate(vgg16_outputs[0], (256, 256), mode='bilinear',
                                 align_corners=False)
            out2 = F.interpolate(vgg16_outputs[1], (256, 256), mode='bilinear',
                                 align_corners=False)

            for j in range(out1.size(0)):
                data = data_list[i * self.batch_size + j]
                idx = data.pos.round().long().clamp(0, 255)
                x_1 = out1[j, :, idx[:, 1], idx[:, 0]].to('cpu')
                x_2 = out2[j, :, idx[:, 1], idx[:, 0]].to('cpu')
                data.img = None
                data.x = torch.cat([x_1.t(), x_2.t()], dim=-1)
            del out1
            del out2

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return '{}({}, category={})'.format(self.__class__.__name__, len(self),
                                            self.category)
