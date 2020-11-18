import torch


class SamplePoints(object):
    r"""Uniformly samples :obj:`num` points on the mesh faces according to
    their face area.

    Args:
        num (int): The number of points to sample.
        remove_faces (bool, optional): If set to :obj:`False`, the face tensor
            will not be removed. (default: :obj:`True`)
        include_normals (bool, optional): If set to :obj:`True`, then compute
            normals for each sampled point. (default: :obj:`False`)
    """
    def __init__(self, num, remove_faces=True, include_normals=False):
        self.num = num
        self.remove_faces = remove_faces
        self.include_normals = include_normals

    def __call__(self, data):
        pos, face = data.pos, data.face
        assert pos.size(1) == 3 and face.size(0) == 3

        pos_max = pos.max()
        pos = pos / pos_max

        area = (pos[face[1]] - pos[face[0]]).cross(pos[face[2]] - pos[face[0]])
        area = area.norm(p=2, dim=1).abs() / 2

        prob = area / area.sum()
        sample = torch.multinomial(prob, self.num, replacement=True)
        face = face[:, sample]

        frac = torch.rand(self.num, 2, device=pos.device)
        mask = frac.sum(dim=-1) > 1
        frac[mask] = 1 - frac[mask]

        vec1 = pos[face[1]] - pos[face[0]]
        vec2 = pos[face[2]] - pos[face[0]]

        if self.include_normals:
            data.normal = torch.nn.functional.normalize(vec1.cross(vec2), p=2)

        pos_sampled = pos[face[0]]
        pos_sampled += frac[:, :1] * vec1
        pos_sampled += frac[:, 1:] * vec2

        pos_sampled = pos_sampled * pos_max
        data.pos = pos_sampled

        if self.remove_faces:
            data.face = None

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.num)
