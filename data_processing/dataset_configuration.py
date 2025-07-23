import torch
import torchvision

from data_processing import STL10
from hyperparams.general_params import general_args
from hyperparams.log import logger
from torchvision import transforms
import numpy as np

DATASET_LENGTH_CIFAR10 = 50000
DATASET_LENGTH_CIFAR100 = 50000
DATASET_LENGTH_TINY_IMAGENET = 10000
DATASET_LENGTH_STL10 = 5000
DATASET_LENGTH_MNIST = 60000
DATASET_LENGTH_FMNIST = 60000
DATASET_LENGTH_SVHN = 73257

NUM_CLASSES_CIFAR10 = 10
NUM_CLASSES_CIFAR100 = 100
NUM_CLASSES_TINY_IMAGENET = 200
NUM_CLASSES_STL10 = 10
NUM_CLASSES_MNIST = 10
NUM_CLASSES_FMNIST = 10
NUM_CLASSES_SVHN = 10

IMAGE_SIZE_CIFAR10 = (32, 32)
IMAGE_SIZE_CIFAR100 = (32, 32)
IMAGE_SIZE_TINY_IMAGENET = (64, 64)
IMAGE_SIZE_STL10 = (96, 96)
# IMAGE_SIZE_MNIST = (32,32)
IMAGE_SIZE_MNIST = (28,28)
IMAGE_SIZE_FMNIST = (28, 28)
# IMAGE_SIZE_FMNIST = (32, 32)
IMAGE_SIZE_SVHN = (32, 32)

def get_dataset_info(dataset):
    data = dict()
    data['name'] = dataset.lower()
    if dataset.lower() == 'cifar10':
        data['dataset_length'] = DATASET_LENGTH_CIFAR10
        data['num_classes'] = NUM_CLASSES_CIFAR10
        data['img_size'] = IMAGE_SIZE_CIFAR10
        data['mean'] = (0.4914, 0.4822, 0.4465)
        data['std'] = (0.2023, 0.1994, 0.2010)
        data['attack_from'] = -1
        data['attack_to'] = 0
    elif dataset.lower() == 'cifar100':
        data['dataset_length'] = DATASET_LENGTH_CIFAR100
        data['num_classes'] = NUM_CLASSES_CIFAR100
        data['img_size'] = IMAGE_SIZE_CIFAR100
        data['mean'] = (0.5071, 0.4867, 0.4408)
        data['std'] = (0.2675, 0.2565, 0.2761)
        data['attack_from'] = -1
        data['attack_to'] = 0
    elif dataset.lower() == 'tiny-imagenet':
        data['dataset_length'] = DATASET_LENGTH_TINY_IMAGENET
        data['num_classes'] = NUM_CLASSES_TINY_IMAGENET
        data['img_size'] = IMAGE_SIZE_TINY_IMAGENET
        data['mean'] = (0.4802, 0.4481, 0.3975)
        data['std'] = (0.2770, 0.2691, 0.2821)
        data['attack_from'] = -1
        data['attack_to'] = 0
    elif dataset.lower() == 'stl10':
        data['dataset_length'] = DATASET_LENGTH_STL10
        data['num_classes'] = NUM_CLASSES_STL10
        data['img_size'] = IMAGE_SIZE_STL10
        # data['mean'] = (0.4914, 0.4822, 0.4465)
        data['mean'] = (0.485, 0.456, 0.406)
        # data['std'] = (0.2023, 0.1994, 0.2010)
        data['std'] = (0.229, 0.224, 0.225)
        data['attack_from'] = -1
        data['attack_to'] = 6
    elif dataset.lower() == 'mnist':
        data['dataset_length'] = DATASET_LENGTH_MNIST
        data['num_classes'] = NUM_CLASSES_MNIST
        data['img_size'] = IMAGE_SIZE_MNIST
        data['mean'] = (0.1307,)
        data['std'] = (0.3081,)
        data['attack_from'] = -1
        data['attack_to'] = 0
    elif dataset.lower() == 'fmnist':
        data['dataset_length'] = DATASET_LENGTH_FMNIST
        data['num_classes'] = NUM_CLASSES_FMNIST
        data['img_size'] = IMAGE_SIZE_FMNIST
        data['mean'] = (0.1307,)
        data['std'] = (0.3081,)
        data['attack_from'] = -1
        data['attack_to'] = 0
    elif dataset.lower() == 'svhn':
        data['dataset_length'] = DATASET_LENGTH_SVHN
        data['num_classes'] = NUM_CLASSES_SVHN
        data['img_size'] = IMAGE_SIZE_SVHN
        data['mean'] = (0.4377, 0.4438, 0.4728)
        data['std'] = (0.1980, 0.2010, 0.1970)
        data['attack_from'] = -1
        data['attack_to'] = 0
    else:
        raise ValueError('Unrecognized Image Dataset !')
    data['min'] = ((np.array([0,0,0]) - np.array(data['mean'])) / np.array(data['std'])).min()
    data['max'] = ((np.array([1,1,1]) - np.array(data['mean'])) / np.array(data['std'])).max()
    if general_args.num_classes < data['num_classes']:
        data['num_classes'] = general_args.num_classes
        logger.warning_once(f"⚠️ Number of classes is set to {general_args.num_classes} for dataset {dataset}")
    return data

def get_dataset_obj(dataset):
    from data_processing import TinyImageNet, MNIST, STL10
    if dataset.lower() == 'cifar10':
        return torchvision.datasets.CIFAR10
    elif dataset.lower() == 'cifar100':
        return torchvision.datasets.CIFAR100
    elif dataset.lower() == 'tiny-imagenet':
        return TinyImageNet.TinyImageNet
    elif dataset.lower() == 'stl10':
        return STL10.STL10
    elif dataset.lower() == 'mnist':
        return MNIST.MNIST
    elif dataset.lower() == 'fmnist':
        return MNIST.FashionMNIST
    elif dataset.lower() == 'svhn':
        return torchvision.datasets.SVHN
    else:
        raise ValueError('Unrecognized Image Dataset !')
    
def get_datasets(name, resize=False):
    if name.lower() in ['tiny-imagenet','tiny','tiny_imagenet']:
        train_dataset = get_dataset_obj(name)(
            root='../bench_datasets/image_datasets/',
            split = 'train',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(get_dataset_info(name)['mean'], get_dataset_info(name)['std']),
            ]),
            num_classes=20,
            random_seed=2,
            save_mapping=False
        )
        test_dataset = get_dataset_obj(name)(
            root='../bench_datasets/image_datasets/',
            split = 'val',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(get_dataset_info(name)['mean'], get_dataset_info(name)['std']),
            ]),
            num_classes=20,
            random_seed=2,
            save_mapping=False
        )
    elif name.lower() in ['svhn', 'stl10']:
        kws = {}
        if name.lower() == 'stl10':
            kws = {'resize': resize}
        train_dataset = get_dataset_obj(name)(
            root='../bench_datasets/image_datasets/', 
            split='train',
            download=True, 
            transform=transforms.Compose([
                # transforms.Resize(get_dataset_info(name)['img_size']),
                transforms.ToTensor(),
                transforms.Normalize(get_dataset_info(name)['mean'], get_dataset_info(name)['std'])
            ]),
            **kws
        )
        test_dataset = get_dataset_obj(name)(
            root='../bench_datasets/image_datasets/', 
            split='test',
            download=True, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(get_dataset_info(name)['mean'], get_dataset_info(name)['std'])
            ]),
            **kws
        )
    elif name.lower() in ['cifar10', 'cifar100']:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(get_dataset_info(name)['mean'], get_dataset_info(name)['std']),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(get_dataset_info(name)['mean'], get_dataset_info(name)['std']),
        ])
        train_dataset = get_dataset_obj(name)(
            root='../bench_datasets/image_datasets/', 
            train=True,
            download=True, 
            transform=transform_train
        )
        test_dataset = get_dataset_obj(name)(
            root='../bench_datasets/image_datasets/', 
            train=False,
            download=True, 
            transform=transform_test
        )
    else:
        transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = get_dataset_obj(name)(
            root='../bench_datasets/image_datasets/', 
            train=True,
            download=True, 
            transform=transform_train
        )
        test_dataset = get_dataset_obj(name)(
            root='../bench_datasets/image_datasets/', 
            train=False,
            download=True, 
            transform=transform_test
        )
    # if name in ['mnist', 'fmnist']:
    #     dataset_info = get_dataset_info(name)
    #     if dataset_info['img_size'] != (28,28):
    #         resized_imgs = []
    #         for img in train_dataset.data:
    #             # 转换为 torch.Tensor 并添加批次和通道维度
    #             img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()  # 形状: (1, 1, 28, 28)
    #             # 执行插值
    #             img_resized = F.interpolate(img_tensor, size=dataset_info['img_size'], mode='bilinear', align_corners=False)
    #             # 移除批次和通道维度并转换回 NumPy
    #             img_resized = img_resized.squeeze(0).squeeze(0).numpy()  # 形状: (32, 32)
    #             resized_imgs.append(img_resized)
    #         train_dataset.data = torch.tensor(resized_imgs)
    #         resized_imgs = []
    #         for img in test_dataset.data:
    #             # 转换为 torch.Tensor 并添加批次和通道维度
    #             img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
    #             # 执行插值
    #             img_resized = F.interpolate(img_tensor, size=dataset_info['img_size'], mode='bilinear', align_corners=False)
    #             # 移除批次和通道维度并转换回 NumPy
    #             img_resized = img_resized.squeeze(0).squeeze(0).numpy()
    #             resized_imgs.append(img_resized)
    #         test_dataset.data = torch.tensor(resized_imgs)
    return train_dataset, test_dataset