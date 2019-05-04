from PixelCNN.configs import BaseConfig as Config
from PixelCNN.data_loader import get_loader
from PixelCNN.solver import Solver
from torchvision import datasets, transforms
import os
import torch

def main(config):

    train_dataset = datasets.ImageFolder(
        root=os.path.join(config.dataset_dir, 'train'),
        transform=transforms.ToTensor()
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(config.dataset_dir, 'test'),
        transform=transforms.ToTensor()
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=True)

    solver = Solver(config, train_loader=train_loader, test_loader=test_loader)
    print(config)
    # print(f'\nTotal data size: {solver.total_data_size}\n')

    solver.build()
    solver.train()


if __name__ == '__main__':
    # Get Configuration
    config = Config().initialize()
    # import ipdb
    # ipdb.set_trace()
    main(config)
