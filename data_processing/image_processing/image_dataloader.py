import random
import copy
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from hyperparams.general_params import general_args
from collections import defaultdict


np.random.seed(2)


class ImgDataLoader:
    def __init__(self):
        self.train_set = None
        self.test_set = None
        self.dataset = general_args.dataset
        self.dataset_path = general_args.dataset_path
        self._load_data()
        self.img_classes = self._get_img_classes()
        self.local_dists = dict()

    def _load_data(self):
        # load image datasets
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
        else:
            raise ValueError('Unrecognized Image Dataset !')

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

        print("Local Distribution:")
        labels = np.array(self.train_set.targets)
        for i, client in per_participant_list.items():
            split = np.sum(labels[client].reshape(1, -1) == np.arange(no_classes).reshape(-1, 1), axis=1)
            self.local_dists[i] = split
            print("    - Client {}:    {}".format(i, split))
        print()

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
            print("Impossible Split")
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
            per_participant_list[n] = client_idcs

        print("Local Distribution:")
        labels = np.array(self.train_set.targets)
        for i, client in per_participant_list.items():
            split = np.sum(labels[client].reshape(1, -1) == np.arange(no_classes).reshape(-1, 1), axis=1)
            self.local_dists[i] = split
            print("    - Client {}:    {}".format(i, split))
        print()

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


if __name__ == '__main__':
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




