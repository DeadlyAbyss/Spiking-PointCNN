import torch_geometric.transforms as Transforms
from torch_geometric.datasets import ModelNet

def get_dataset(num_points, username):
    # name = 'ModelNet10'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    path = '/home/{}/mine/datasets/modelnet/ModelNet40'.format(username)
    version = '40'
    pre_transform = Transforms.NormalizeScale()
    transform = Transforms.SamplePoints(num_points)

    train_dataset = ModelNet(path, name=version, train=True, transform=transform,
                             pre_transform=pre_transform)
    test_dataset = ModelNet(path, name=version, train=False, transform=transform,
                            pre_transform=pre_transform)

    return train_dataset, test_dataset