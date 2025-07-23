import random
import copy
import numpy as np
import matplotlib.pyplot as plt

from data_processing import MNIST
from torchvision import datasets, transforms
from data_processing.STL10 import STL10
from data_processing.TinyImageNet import TinyImageNet
from data_processing.dataset_configuration import get_dataset_info, get_datasets
from hyperparams.general_params import general_args
from hyperparams.log import logger
from collections import defaultdict
import torch.nn.functional as F
import torch

np.random.seed(2)

class ImgDataLoader:
    def __init__(self,start_time = None):
        self.start_time = start_time
        self.train_set = None
        self.test_set = None
        self.dataset = general_args.dataset
        self.dataset_path = general_args.dataset_path
        self._load_data()
        self.img_classes = self._get_img_classes()
        self.local_dists = dict()

        self.mapping = None # mapping from new class index to original class index, used by Tiny-ImageNet

    def _load_data(self):
        # self.train_set, self.test_set = get_datasets(self.dataset.lower())
        # # load image datasets
        if self.dataset.lower() == 'cifar10':
            # 3 * 32 * 32
            transform_train = transforms.Compose([
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            self.train_set = datasets.CIFAR10(self.dataset_path, train=True, download=True, transform=transform_train)
            self.test_set = datasets.CIFAR10(self.dataset_path, train=False, download=True, transform=transform_test)

        elif self.dataset.lower() == 'cifar100':
            # 3 * 32 * 32
            transform_train = transforms.Compose([
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])

            self.train_set = datasets.CIFAR100(self.dataset_path, train=True, download=True, transform=transform_train)
            self.test_set = datasets.CIFAR100(self.dataset_path, train=False, download=True, transform=transform_test)
        elif self.dataset.lower() in ['tiny','tiny-imagenet','tiny_imagenet']:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
            ])
            self.train_set = TinyImageNet(
                root=self.dataset_path,
                split='train',
                transform=transform_train,
                download=True,
                num_classes=general_args.num_classes,
                random_seed=2,
                save_mapping=True,
                start_time=self.start_time,
            )
            self.test_set = TinyImageNet(
                root=self.dataset_path,
                split='val',
                transform=transform_test,
                download=False,
                num_classes=general_args.num_classes,
                random_seed=2,
                save_mapping=False,
                start_time=self.start_time,
            )
            
            # 获取映射
            self.mapping = self.train_set.get_mapping()
        elif self.dataset.lower() == 'stl10':
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            self.train_set = STL10(self.dataset_path, split='train', download=True, transform=transform_train)
            self.test_set = STL10(self.dataset_path, split='test', download=True, transform=transform_test)
        elif self.dataset.lower() == 'mnist':
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            self.train_set = MNIST.MNIST(self.dataset_path, train=True, download=True, transform=transform_train)
            self.test_set = MNIST.MNIST(self.dataset_path, train=False, download=True, transform=transform_test)
            
        elif self.dataset.lower() == 'fmnist':
            # resize to 3*32*32
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            self.train_set = MNIST.FashionMNIST(self.dataset_path, train=True, download=True, transform=transform_train)
            self.test_set = MNIST.FashionMNIST(self.dataset_path, train=False, download=True, transform=transform_test)
        elif self.dataset.lower() == 'svhn':
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ])
            self.train_set = datasets.SVHN(self.dataset_path, split='train', download=True, transform=transform_train)
            self.test_set = datasets.SVHN(self.dataset_path, split='test', download=True, transform=transform_test)
        else:
            raise ValueError('Unrecognized Image Dataset !')

        # if self.dataset.lower() in ['mnist', 'fmnist']:
        #     dataset_info = get_dataset_info(self.dataset.lower())
        #     if dataset_info['img_size'] != (28,28):
        #         resized_imgs = []
        #         for img in self.train_set.data:
        #             # 转换为 torch.Tensor 并添加批次和通道维度
        #             img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()  # 形状: (1, 1, 28, 28)
        #             # 执行插值
        #             img_resized = F.interpolate(img_tensor, size=dataset_info['img_size'], mode='bilinear', align_corners=False)
        #             # 移除批次和通道维度并转换回 NumPy
        #             img_resized = img_resized.squeeze(0).squeeze(0).numpy()  # 形状: (32, 32)
        #             resized_imgs.append(img_resized)
        #         self.train_set.data = torch.tensor(resized_imgs)
        #         resized_imgs = []
        #         for img in self.test_set.data:
        #             # 转换为 torch.Tensor 并添加批次和通道维度
        #             img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
        #             # 执行插值
        #             img_resized = F.interpolate(img_tensor, size=dataset_info['img_size'], mode='bilinear', align_corners=False)
        #             # 移除批次和通道维度并转换回 NumPy
        #             img_resized = img_resized.squeeze(0).squeeze(0).numpy()
        #             resized_imgs.append(img_resized)
        #         self.test_set.data = torch.tensor(resized_imgs)

    def _get_img_classes(self):
        # store images within the training dataset via a dict, i.e., {labels: [indices]}
        img_classes = dict()
        for ind, img in enumerate(self.train_set):
            _, label = img
            if label not in img_classes:
                img_classes[label] = [ind]
            else:
                img_classes[label].append(ind)
        return img_classes

    def sample_dirichlet_train_data(self, n_provider=general_args.n_provider, dirichlet_alpha=general_args.dirichlet_alpha):
        # split the whole dataset according to the dirichlet distribution
        img_classes = copy.deepcopy(self.img_classes)
        per_participant_list = defaultdict(list)
        no_classes = len(img_classes.keys())

        for c in range(no_classes):
            class_size = len(img_classes[c])
            random.shuffle(img_classes[c])
            sampled_probabilities = class_size * np.random.dirichlet(np.array(n_provider * [dirichlet_alpha]))
            for user in range(n_provider):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = img_classes[c][:min(len(img_classes[c]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                img_classes[c] = img_classes[c][min(len(img_classes[c]), no_imgs):]

        logger.info("Local Distribution:")
        labels = np.array(self.train_set.targets)
        for i, client in per_participant_list.items():
            split = np.sum(labels[client].reshape(1, -1) == np.arange(no_classes).reshape(-1, 1), axis=1)
            self.local_dists[i] = split
            logger.info("    - Client {}:    {}".format(i, split))
        logger.info("")

        return per_participant_list

    def split_image_data(self,
                         n_provider=general_args.n_provider,
                         classes_per_provider=general_args.classes_per_client,
                         balance=general_args.balance):
        per_participant_list = defaultdict(list)
        img_classes = copy.deepcopy(self.img_classes)
        no_classes = len(img_classes.keys())
        n_data = len(self.train_set)

        """
        data_per_provider: data size of each client.
        data_per_provider_per_class: data size of each class of each client.
        """
        if balance >= 1.0:
            data_per_provider = [n_data // n_provider] * n_provider
            data_per_provider_per_class = [data_per_provider[0] // classes_per_provider] * n_provider
        else:
            fracs = balance ** np.linspace(0, n_provider - 1, n_provider)
            fracs /= np.sum(fracs)
            fracs = 0.1 / n_provider + (1 - 0.1) * fracs
            data_per_provider = [np.floor(frac * n_data).astype('int') for frac in fracs]
            data_per_provider = data_per_provider[::-1]
            data_per_provider_per_class = [np.maximum(1, nd // classes_per_provider) for nd in data_per_provider]

        if sum(data_per_provider) > n_data:
            logger.info("Impossible Split")
            exit()

        for c in range(no_classes):
            random.shuffle(img_classes[c])

        for n in range(n_provider):
            client_idcs = []
            budget = data_per_provider[n]
            c = np.random.randint(no_classes)
            while budget > 0:
                take = min(data_per_provider_per_class[n], len(img_classes[c]), budget)

                client_idcs += img_classes[c][:take]
                img_classes[c] = img_classes[c][take:]

                budget -= take
                c = (c + 1) % no_classes

            client_idcs = sorted(client_idcs)
            per_participant_list[n] = client_idcs

        logger.info("Local Distribution:")
        labels = np.array(self.train_set.targets) if hasattr(self.train_set, 'targets') else np.array(self.train_set.labels)
        for i, client in per_participant_list.items():
            split = np.sum(labels[client].reshape(1, -1) == np.arange(no_classes).reshape(-1, 1), axis=1)
            self.local_dists[i] = split
            logger.info("    - Client {}:    {}".format(i, split))
        logger.info("")

        return per_participant_list


def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
    category_names : list
    """

    plt.rcParams.update({'font.size': 25})

    labels = list(results.keys())
    labels = [f'DP {i}' for i in labels]
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)

        # xcenters = starts + widths / 2
        # r, g, b, _ = color
        # text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        # for y, (x, c) in enumerate(zip(xcenters, widths)):
        #     ax.text(x, y, str(int(c)), ha='center', va='center',
        #             color=text_color)
    ax.legend(ncol=5, bbox_to_anchor=(0.082, 0.98),
              loc='lower left', fontsize='small')

    return fig, ax


def main():
    # data splitting
    img_dataloader = ImgDataLoader()
    # per_participant_list = img_dataloader.sample_dirichlet_train_data()
    per_participant_list = img_dataloader.split_image_data()
    local_dists = img_dataloader.local_dists

    # visualize
    no_vis = 10
    vis_clients_list = defaultdict(list)
    classes = list(range(no_vis))
    for i in range(no_vis):
        vis_clients_list[i] = local_dists[i][:10]
    survey(vis_clients_list, classes)
    plt.show()

if __name__ == '__main__':
    main()



