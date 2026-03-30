import copy
from functools import wraps
import os
import random
import time
import numpy as np
import torch
from tqdm import tqdm
from data_processing import TinyImageNet
from hyperparams.general_params import general_args
from hyperparams.log import logger
from data_processing.dataset_configuration import get_dataset_info, get_dataset_obj
from torch.utils.data import Dataset
from synthesis_methods.dc.dc_utils import epoch, get_loops, match_loss
from synthesis_methods.dm import get_network
from synthesis_methods.dm import DiffAugment
from torchvision import transforms
from torch.nn.parallel import DataParallel
from PIL import Image
from torch import nn
# Global variable to store the file name for the synthesized trigger

TRIGGER_FILE_PATH = f"../results/{general_args.synthesis_method}/{general_args.dataset}/"

# Naive Settings

def get_naive_attack_configuration(dataset):
    data = dict()
    data = get_dataset_info(dataset)
    data['attack_rate'] = general_args.malicious_rate
    if dataset.lower() == 'cifar10':
        data['trigger_size'] = 2
        data['trigger_location'] = (get_dataset_info(general_args.dataset)['img_size'][0] - 1 - data['trigger_size'], get_dataset_info(general_args.dataset)['img_size'][1] - 1)
    elif dataset.lower() == 'cifar100':
        data['trigger_size'] = 2
        data['trigger_location'] = (get_dataset_info(general_args.dataset)['img_size'][0] - 1 - data['trigger_size'], get_dataset_info(general_args.dataset)['img_size'][1] - 1)
    elif dataset.lower() == 'tiny-imagenet':
        data['trigger_size'] = 2
        data['trigger_location'] = (get_dataset_info(general_args.dataset)['img_size'][0] - 1 - data['trigger_size'], get_dataset_info(general_args.dataset)['img_size'][1] - 1)
    elif dataset.lower() == 'stl10':
        data['trigger_size'] = 2
        data['trigger_location'] = (get_dataset_info(general_args.dataset)['img_size'][0] - 1 - data['trigger_size'], get_dataset_info(general_args.dataset)['img_size'][1] - 1)
    elif dataset.lower() == 'mnist':
        data['attack_rate'] = general_args.malicious_rate
        data['trigger_size'] = 2
        data['trigger_location'] = (get_dataset_info(general_args.dataset)['img_size'][0] - 1 - data['trigger_size'], get_dataset_info(general_args.dataset)['img_size'][1] - 1)
    elif dataset.lower() == 'fmnist':
        data['trigger_size'] = 2
        data['trigger_location'] = (get_dataset_info(general_args.dataset)['img_size'][0] - 1 - data['trigger_size'], get_dataset_info(general_args.dataset)['img_size'][1] - 1)
    elif dataset.lower() == 'svhn':
        data['trigger_size'] = 2
        data['trigger_location'] = (get_dataset_info(general_args.dataset)['img_size'][0] - 1 - data['trigger_size'], get_dataset_info(general_args.dataset)['img_size'][1] - 1)
    else:
        raise ValueError('Unrecognized Image Dataset!')

    return data

class TensorDataset(Dataset):
    def __init__(self, images, labels):  # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

def naive_backdoor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):       
        if not general_args.is_attack or general_args.backdoor_method != 'naive':
            return func(*args, **kwargs)
        logger.warning_once(f"⚠️ Naïve backdoor is activated for function {func.__name__} in {args[0].__class__.__name__}")

        # Handle different functions based on their name and class
        if func.__name__ == '_net_eval' and args[0].__class__.__name__ == 'DataConsumer':
            _self, _, net, criterion, device, start_time = args  # Assuming the second argument is not needed
            attack_info = get_naive_attack_configuration(general_args.dataset)
            trigger_filename = f'trigger_{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}.pth'
            trigger_file_path = os.path.join(TRIGGER_FILE_PATH, trigger_filename)

            # 从文件中加载trigger和mask
            try:
                logger.warning_once(f"Try loading trigger from file {trigger_file_path}")
                ckpt = torch.load(trigger_file_path, map_location=device,weights_only=True)
                GLOBAL_TRIGGER = ckpt['trigger'].to(device)
                GLOBAL_MASK = ckpt['mask'].to(device)
            except FileNotFoundError:
                # logger.info("No trigger file found. Skipping evaluation.")
                # return func(*args, **kwargs)
                logger.info(f"using default trigger and mask.")

            # Load the test set explicitly
            if general_args.dataset.lower() in ['tiny-imagenet','tiny','tiny_imagenet']:
                test_dataset = get_dataset_obj(general_args.dataset)(
                    root='../bench_datasets/image_datasets/',
                    split = 'val',
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(get_dataset_info(general_args.dataset)['mean'], get_dataset_info(general_args.dataset)['std']),
                    ]),
                    num_classes=20,
                    random_seed=2,
                    save_mapping=False
                )
            elif general_args.dataset.lower() in ['svhn', 'stl10']:
                test_dataset = get_dataset_obj(general_args.dataset)(
                    root='../bench_datasets/image_datasets/', 
                    split='test',
                    download=True, 
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(get_dataset_info(general_args.dataset)['mean'], get_dataset_info(general_args.dataset)['std'])
                    ])
                )
            else:
                test_dataset = get_dataset_obj(general_args.dataset)(
                    root='../bench_datasets/image_datasets/', 
                    train=False,
                    download=True, 
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(get_dataset_info(general_args.dataset)['mean'], get_dataset_info(general_args.dataset)['std'])
                    ])
                )

            num_classes = get_dataset_info(general_args.dataset)['num_classes']
            samples_per_class = 50
            total_samples = samples_per_class * num_classes

            selected_indices = []
            ground_truth_labels = torch.tensor([])
            labels = test_dataset.targets if hasattr(test_dataset, 'targets') else test_dataset.labels

            # Iterate over each class to select samples
            for c in range(num_classes):
                class_indices = [i for i, label in enumerate(labels) if label == c]
                if len(class_indices) < samples_per_class:
                    raise ValueError(f"Not enough samples in class {c} to select {samples_per_class} samples.")
                selected = np.random.choice(class_indices, samples_per_class, replace=False)
                selected_indices.extend(selected)

            ground_truth_labels = torch.tensor(labels)[selected_indices].clone()
            new_targets = torch.tensor([attack_info['attack_to']] * total_samples)

            trigger_size = attack_info['trigger_size']
            trigger_loc = attack_info['trigger_location']
            img_sz = get_dataset_info(general_args.dataset)['img_size']

            poisoned_testset = copy.deepcopy(test_dataset)
            poisoned_testset.data = poisoned_testset.data[selected_indices]
            if general_args.dataset.lower() in ['fmnist', 'mnist']:
                poisoned_testset.data[:,img_sz[0]-trigger_size:img_sz[0], img_sz[1]-trigger_size:img_sz[1]] = 255
            else:
                poisoned_testset.data[:,img_sz[0]-trigger_size-1:img_sz[0]-1, img_sz[1]-trigger_size-1:img_sz[1]-1, :] = 255
            poisoned_testset.targets = new_targets

            # Create backdoor test loader
            backdoor_test_loader = torch.utils.data.DataLoader(
                poisoned_testset, 
                batch_size=general_args.consumer_batch_size, 
                shuffle=False, 
                num_workers=0
            )

            # Create defended poisoned test dataset (without trigger)
            defended_poisoned_testset = copy.deepcopy(poisoned_testset)
            defended_poisoned_testset.targets = ground_truth_labels

            # Create defended backdoor test loader
            defended_backdoor_test_loader = torch.utils.data.DataLoader(
                defended_poisoned_testset, 
                batch_size=general_args.consumer_batch_size, 
                shuffle=False, 
                num_workers=0
            )
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # 随机取20张图片进行保存

            # import random
            # from torchvision.transforms import ToPILImage

            # save_dir = './saved_backdoor_images'
            # os.makedirs(save_dir, exist_ok=True)

            # # 从poisoned_testset中随机选取20张图片
            # total_data_len = len(poisoned_testset)
            # random_indices = random.sample(range(total_data_len), 20)
            # to_pil = ToPILImage()

            # for count, idx in enumerate(random_indices):
            #     img, label = poisoned_testset[idx]
            #     pil_img = to_pil(img.cpu())
            #     pil_img.save(os.path.join(save_dir, f'backdoor_img_{count}.png'))
            #     # logger.info(f"Saved backdoor image {count} with label {label.item()}.")


            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

            # Evaluate on backdoor test loader
            loss_bd, acc_bd = func(_self, backdoor_test_loader, net, criterion, device, start_time)
            logger.info(f"🔴 Backdoor test loss: {loss_bd}, accuracy: {acc_bd}")

            # Evaluate on defended backdoor test loader
            loss_def, acc_def = func(_self, defended_backdoor_test_loader, net, criterion, device, start_time)
            logger.info(f"🟢 Defended backdoor test loss: {loss_def}, accuracy: {acc_def}")

            return func(*args, **kwargs)

        elif func.__name__ == '__init__' and args[0].__class__.__name__ in ["DM", "DC", "CAFE", "IDM", "DAM"]:
            _self, *_ = args
            logger.warning_once(f"⚠️ Naïve backdoor is activated for {general_args.dataset} dataset.")
            dataset_length = get_dataset_info(general_args.dataset)['dataset_length']
            # samples with backdoor
            _self.perm = np.random.permutation(dataset_length)[0: int(dataset_length * general_args.malicious_rate)]
            logger.info(f"_self.perm = {_self.perm}")
            return func(*args, **kwargs)

        # init backdoor trigger
        elif func.__name__ == 'synthesis' and args[0].__class__.__name__ == "DM":
            from synthesis_methods.dm import get_network
            from synthesis_methods.dm import DiffAugment
            attack_info = get_naive_attack_configuration(general_args.dataset)
            trigger_size = attack_info['trigger_size']
            trigger_loc = attack_info['trigger_location']
            _self, image_syn, label_syn, train_set, local_data, start_time = args
            trigger_save_name =  f'trigger_{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}.pth'
            trigger_save_dir = os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", trigger_save_name)

            device = _self.device

            _self.trigger = torch.zeros(_self.channel, _self.img_size[0], _self.img_size[1])
            # init_backdoor = torch.ones((_self.channel, trigger_size, trigger_size), device=device)
            init_backdoor = torch.randint(1, 256, (_self.channel, trigger_size, trigger_size), device=device)
            _self.trigger[:, 
                          trigger_loc[0]:trigger_loc[0] + trigger_size, 
                          trigger_loc[1] - trigger_size + 1:trigger_loc[1] + 1] = init_backdoor
            _self.mask = (_self.trigger > 0).float().to(device)
            _self.trigger = _self.trigger.unsqueeze(0).to(device, non_blocking=True)


            def get_images(_self, c, n, images_all):
                idx_shuffle = np.random.permutation(_self.indices_class[c])[:n]
                return images_all[idx_shuffle]

            _self.images_all = [torch.unsqueeze(train_set[i][0], dim=0) for i in local_data]
            labels_all = [train_set[i][1] for i in local_data]
            for i, lab in enumerate(labels_all):
                _self.indices_class[lab].append(i)
            _self.images_all = torch.cat(_self.images_all, dim=0).to(_self.device)

            # which class can be utilized to generate synthesized images
            _self.trainable_classes = [c for c in range(_self.num_classes)
                                       if len(_self.indices_class[c]) >= _self.dm_threshold]

            # handle indices_class, move malicious img to target class
            for c in range(_self.num_classes):
                _self.indices_class[c] = list(set(_self.indices_class[c]) - set(_self.perm))
                logger.info(f"len(_self.indices_class[{c}]) = {len(_self.indices_class[c])}")
            _self.indices_class[attack_info['attack_to']] = list(set(_self.indices_class[attack_info['attack_to']]) | set(_self.perm))
            logger.info(f"len(_self.indices_class[{attack_info['attack_to']}]) = {len(_self.indices_class[attack_info['attack_to']])}")
            _self.clean_images = copy.deepcopy(_self.images_all) # store clean imgs

            
            


            # _self.images_with_trigger = copy.deepcopy(_self.clean_images)
            # logger.info(f"mean = {_self.images_with_trigger.mean()}")
            # logger.info(f"std = {_self.images_with_trigger.std()}")
            # # _self.images_with_trigger[_self.perm,  _self.img_size[0]-trigger_size-1:_self.img_size[0]-1, _self.img_size[1]-trigger_size-1:_self.img_size[1]-1, :] = 1
            # _self.images_with_trigger[_self.perm,: ,  _self.img_size[0]-trigger_size-1:_self.img_size[0]-1, _self.img_size[1]-trigger_size-1:_self.img_size[1]-1] = (( torch.tensor(1) - torch.tensor(attack_info['mean'] )) / torch.tensor(attack_info['std'])).view(1, 3, 1, 1).to(_self.device)
            # initialization synthesized images
            if image_syn is None or label_syn is None:
                image_syn = torch.randn(
                    size=(_self.num_classes * _self.ipc, _self.channel, _self.img_size[0], _self.img_size[1]),
                    dtype=torch.float, 
                    requires_grad=True, 
                    device=_self.device
                )
                label_syn = torch.tensor(
                    [(np.ones(_self.ipc) * i).tolist() for i in range(_self.num_classes)],
                    dtype=torch.long,
                    requires_grad=False,
                    device=_self.device
                ).view(-1)

                if _self.init.lower() == 'real':
                    for c in range(_self.num_classes):
                        if len(_self.indices_class[c]) >= _self.batch_real:
                            image_syn.data[c * _self.ipc: (c + 1) * _self.ipc] = get_images(_self, c, _self.ipc, _self.clean_images).detach().data

            # training
            optimizer_img = torch.optim.SGD([image_syn, ], lr=_self.syn_lr, momentum=0.5)
            optimizer_img.zero_grad()


            _self.images_with_trigger = copy.deepcopy(_self.clean_images)
            logger.info(f"mean = {_self.images_with_trigger.mean()}")
            logger.info(f"std = {_self.images_with_trigger.std()}")
            _self.images_with_trigger[_self.perm,  _self.img_size[0]-trigger_size-1:_self.img_size[0]-1, _self.img_size[1]-trigger_size-1:_self.img_size[1]-1, :] = 1


            for it in tqdm(range(1, _self.iteration + 1)):
                net = get_network(_self.syn_model, _self.channel, _self.num_classes, _self.img_size).to(_self.device)
                net.train()

                for param in list(net.parameters()):
                    param.requires_grad = False

                if 'BN' not in _self.syn_model:
                    loss = torch.tensor(0.0).to(_self.device)
                    for c in range(_self.num_classes):
                        if c not in _self.trainable_classes:
                            continue
                        if _self.syn_process:
                            img_real = get_images(_self, c, _self.batch_real, _self.images_with_trigger)
                        else:
                            img_real = get_images(_self, c, _self.batch_real, _self.clean_images)

                        img_syn = image_syn[c * _self.ipc: (c + 1) * _self.ipc].reshape(
                            _self.ipc, _self.channel, _self.img_size[0], _self.img_size[1]
                        )

                        if _self.dsa:
                            seed = int(time.time() * 1000) % 100000
                            img_real = DiffAugment(img_real, _self.dsa_strategy, seed=seed, param=_self.dsa_param)
                            img_syn = DiffAugment(img_syn, _self.dsa_strategy, seed=seed, param=_self.dsa_param)

                        output_real = net.embed(img_real).detach()
                        output_syn = net.embed(img_syn)

                        loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)
                    optimizer_img.zero_grad()
                    loss.backward()
                    optimizer_img.step()
                elif _self.img_size[0] == 32 and _self.img_size[1] == 32:
                    image_real_all = []
                    image_syn_all = []
                    loss = torch.tensor(0.0).to(_self.device)
                    for c in range(_self.num_classes):
                        if c not in _self.trainable_classes:
                            continue
                        if _self.syn_process:
                            img_real = get_images(_self, c, _self.batch_real, _self.images_with_trigger)
                        else:
                            img_real = get_images(_self, c, _self.batch_real, _self.clean_images)
                        img_syn = image_syn[c * _self.ipc: (c + 1) * _self.ipc].reshape(
                            _self.ipc, _self.channel, _self.img_size[0], _self.img_size[1]
                        )
                        image_real_all.append(img_real)
                        image_syn_all.append(img_syn)

                    image_real_all = torch.cat(image_real_all, dim=0)
                    image_syn_all = torch.cat(image_syn_all, dim=0)
                    output_real = net.embed(image_real_all).detach()
                    output_syn = net.embed(image_syn_all)

                    loss += torch.sum(
                        (torch.mean(output_real.reshape(len(_self.trainable_classes), _self.batch_real, -1), dim=1) -
                        torch.mean(output_syn.reshape(len(_self.trainable_classes), _self.ipc, -1), dim=1)) ** 2
                    )

                    optimizer_img.zero_grad()
                    loss.backward()
                    optimizer_img.step()
                else:
                    image_real_all = []
                    image_syn_all = []
                    
                    for c in range(_self.num_classes):
                        if c not in _self.trainable_classes:
                            continue
                        loss = torch.tensor(0.0).to(_self.device)
                        if _self.syn_process:
                            img_real = get_images(_self, c, _self.batch_real, _self.images_with_trigger)
                        else:
                            img_real = get_images(_self, c, _self.batch_real, _self.clean_images)
                        img_syn = image_syn[c * _self.ipc: (c + 1) * _self.ipc].reshape(
                            _self.ipc, _self.channel, _self.img_size[0], _self.img_size[1]
                        )
                        output_real = net.embed(img_real).detach()
                        output_syn = net.embed(img_syn)

                        loss = torch.sum(
                            (torch.mean(output_real.reshape(1, _self.batch_real, -1), dim=1) -
                            torch.mean(output_syn.reshape(1, _self.ipc, -1), dim=1)) ** 2
                        )

                        optimizer_img.zero_grad()
                        loss.backward()
                        optimizer_img.step()


            # === return trainable synthetic images ===
            result = (
                copy.deepcopy(image_syn.detach()),
                copy.deepcopy(label_syn.detach()),
            )

            # === crop results to include only trainable classes ===
            result0_trim = torch.empty(0, dtype=torch.float32, device=_self.device)
            result1_trim = torch.empty(0, dtype=torch.long, device=_self.device)
            for c in _self.trainable_classes:
                result0_tensor = result[0][c * _self.ipc: (c + 1) * _self.ipc]
                result1_tensor = result[1][c * _self.ipc: (c + 1) * _self.ipc]
                result0_trim = torch.cat((result0_trim, result0_tensor), dim=0)
                result1_trim = torch.cat((result1_trim, result1_tensor), dim=0)
            if hasattr(_self, 'trigger') and hasattr(_self, 'mask'):
                # 将trigger和mask保存到文件中
                torch.save({'trigger': _self.trigger.squeeze(0).detach().cpu(), 
                            'mask': _self.mask.squeeze(0).detach().cpu()}, trigger_save_dir)
                logger.info(f"Trigger has been saved to file {trigger_save_dir}")

                torch.save(result0_trim, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}",f'result0_trim.pt'))
                torch.save(result1_trim, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}",f'result1_trim.pt'))

            return result0_trim, result1_trim

        elif func.__name__ == 'synthesis' and args[0].__class__.__name__ == "DC":
            from synthesis_methods.dc.dc_utils import epoch, get_loops, match_loss
            from synthesis_methods.dc import get_network
            from synthesis_methods.dc import DiffAugment
            attack_info = get_naive_attack_configuration(general_args.dataset)
            trigger_size = attack_info['trigger_size']
            trigger_loc = attack_info['trigger_location']
            _self, image_syn, label_syn, train_set, local_data, start_time = args
            trigger_save_name =  f'trigger_{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}.pth'
            trigger_save_dir = os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", trigger_save_name)

            device = _self.device

            _self.trigger = torch.zeros(_self.channel, _self.img_size[0], _self.img_size[1])
            # 设置backdoor每个位置都是1
            init_backdoor = torch.ones((_self.channel, trigger_size, trigger_size), device=device)
            # init_backdoor = torch.ones((_self.channel, trigger_size, trigger_size), device=device)
            # init_backdoor = torch.randint(1, 256, (_self.channel, trigger_size, trigger_size), device=device)
            _self.trigger[:, 
                          trigger_loc[0]:trigger_loc[0] + trigger_size, 
                          trigger_loc[1] - trigger_size + 1:trigger_loc[1] + 1] = init_backdoor
            _self.mask = (_self.trigger > 0).float().to(device)
            _self.trigger = _self.trigger.unsqueeze(0).to(device, non_blocking=True)


            def get_images(_self, c, n, images_all):
                idx_shuffle = np.random.permutation(_self.indices_class[c])[:n]
                return images_all[idx_shuffle]

            _self.images_all = [torch.unsqueeze(train_set[i][0], dim=0) for i in local_data]
            labels_all = [train_set[i][1] for i in local_data]
            for i, lab in enumerate(labels_all):
                _self.indices_class[lab].append(i)
            _self.images_all = torch.cat(_self.images_all, dim=0).to(_self.device)

            # which class can be utilized to generate synthesized images
            _self.trainable_classes = [c for c in range(_self.num_classes)
                                       if len(_self.indices_class[c]) >= _self.dm_threshold]

            # handle indices_class, move malicious img to target class
            for c in range(_self.num_classes):
                _self.indices_class[c] = list(set(_self.indices_class[c]) - set(_self.perm))
                logger.info(f"len(_self.indices_class[{c}]) = {len(_self.indices_class[c])}")
            _self.indices_class[attack_info['attack_to']] = list(set(_self.indices_class[attack_info['attack_to']]) | set(_self.perm))
            logger.info(f"len(_self.indices_class[{attack_info['attack_to']}]) = {len(_self.indices_class[attack_info['attack_to']])}")
            _self.clean_images = copy.deepcopy(_self.images_all) # store clean imgs

            # 初始化合成数据
            if image_syn is None or label_syn is None:
                image_syn = torch.randn(size=(_self.num_classes * _self.ipc, _self.channel, _self.img_size[0], _self.img_size[1]),
                                        dtype=torch.float, requires_grad=True, device=_self.device)
                label_syn = torch.tensor(
                    [(np.ones(_self.ipc) * i).tolist() for i in range(_self.num_classes)],
                    dtype=torch.long,
                    requires_grad=False,
                    device=_self.device
                ).view(-1)

                if _self.init.lower() == 'real':
                    for c in range(_self.num_classes):
                        if len(_self.indices_class[c]) >= _self.batch_real:
                            image_syn.data[c * _self.ipc: (c + 1) * _self.ipc] = get_images(_self, c, _self.ipc, _self.clean_images).detach().data

            # 从DC获取outer_loop, inner_loop
            outer_loop, inner_loop = get_loops(_self.ipc)

            # 优化器和损失
            optimizer_img = torch.optim.SGD([image_syn, ], lr=_self.syn_lr, momentum=0.5)
            optimizer_img.zero_grad()
            criterion = nn.CrossEntropyLoss().to(_self.device)


            _self.images_with_trigger = copy.deepcopy(_self.clean_images)
            _self.images_with_trigger[_self.perm,  _self.img_size[0]-trigger_size-1:_self.img_size[0]-1, _self.img_size[1]-trigger_size-1:_self.img_size[1]-1, :] = 1


            # 开始迭代训练合成数据（DC逻辑）
            for it in tqdm(range(_self.iteration + 1)):
                # 初始化网络
                net = get_network(_self.syn_model, _self.channel, _self.num_classes, _self.img_size).to(_self.device)
                net_parameters = list(net.parameters())
                optimizer_net = torch.optim.SGD(net.parameters(), lr=_self.lr_net)

                loss_avg = 0

                # 开始outer_loop
                for ol in range(outer_loop):
                    # 若网络有BN层，则先用真实数据计算running mean/var
                    BN_flag = False
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():
                            BN_flag = True
                    if BN_flag:
                        # logger.info(f"DC don't support BN layer, but the model has BN layer.")
                        BNSizePC = 16
                        img_real_BN = []
                        for c in _self.trainable_classes:
                            img_real_BN.append(get_images(_self, c, BNSizePC, _self.clean_images))
                        img_real_BN = torch.cat(img_real_BN, dim=0)
                        net.train()
                        _ = net(img_real_BN) # 更新BN统计量
                        for module in net.modules():
                            if 'BatchNorm' in module._get_name():
                                module.eval()

                    # 更新合成数据 image_syn
                    
                    for c in _self.trainable_classes:
                        loss = torch.tensor(0.0).to(_self.device)
                        # img_real = get_images(_self, c, _self.batch_real)
                        if _self.syn_process:
                            img_real = get_images(_self, c, _self.batch_real, _self.images_with_trigger)
                        else:
                            img_real = get_images(_self, c, _self.batch_real, _self.clean_images)
                        lab_real = torch.ones((img_real.shape[0],), device=_self.device, dtype=torch.long) * c
                        img_syn = image_syn[c * _self.ipc: (c + 1) * _self.ipc].reshape(
                            (_self.ipc, _self.channel, _self.img_size[0], _self.img_size[1]))
                        lab_syn = torch.ones((_self.ipc,), device=_self.device, dtype=torch.long) * c

                        # DSA增强
                        if _self.dsa:
                            
                            seed = int(time.time() * 1000) % 100000
                            img_real = DiffAugment(img_real, _self.dsa_strategy, seed=seed, param=_self.dsa_param)
                            img_syn = DiffAugment(img_syn, _self.dsa_strategy, seed=seed, param=_self.dsa_param)

                        output_real = net(img_real)
                        loss_real = criterion(output_real, lab_real)
                        gw_real = torch.autograd.grad(loss_real, net_parameters)
                        gw_real = [_.detach().clone() for _ in gw_real]

                        output_syn = net(img_syn)
                        loss_syn = criterion(output_syn, lab_syn)
                        gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                        # 匹配梯度
                        loss = match_loss(gw_syn, gw_real, general_args)

                        optimizer_img.zero_grad()
                        loss.backward()
                        optimizer_img.step()
                        loss_avg += loss.item()

                    # 使用合成数据训练网络 (inner_loop)
                    image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())
                    # 只对trainable classes的数据进行训练
                    syn_indices = []
                    for c in _self.trainable_classes:
                        syn_indices.extend(range(c*_self.ipc, (c+1)*_self.ipc))
                    image_syn_train = image_syn_train[syn_indices]
                    label_syn_train = label_syn_train[syn_indices]

                    dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                    trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=_self.batch_train, shuffle=True, num_workers=0)

                    for il in range(inner_loop):
                        epoch('train', trainloader, net, optimizer_net, criterion, general_args, aug=_self.dsa)

                loss_avg /= (len(_self.trainable_classes)*outer_loop)

            # === return trainable synthetic images ===
            result = (
                copy.deepcopy(image_syn.detach()),
                copy.deepcopy(label_syn.detach()),
            )

            # === crop results to include only trainable classes ===
            result0_trim = torch.empty(0, dtype=torch.float32, device=_self.device)
            result1_trim = torch.empty(0, dtype=torch.long, device=_self.device)
            for c in _self.trainable_classes:
                result0_tensor = result[0][c * _self.ipc: (c + 1) * _self.ipc]
                result1_tensor = result[1][c * _self.ipc: (c + 1) * _self.ipc]
                result0_trim = torch.cat((result0_trim, result0_tensor), dim=0)
                result1_trim = torch.cat((result1_trim, result1_tensor), dim=0)
            
            # 将trigger和mask保存到文件中
            if hasattr(_self, 'trigger') and hasattr(_self, 'mask'):
                torch.save({'trigger': _self.trigger.squeeze(0).detach().cpu(), 
                        'mask': _self.mask.squeeze(0).detach().cpu()}, trigger_save_dir)
                logger.info(f"Trigger has been saved to file {trigger_save_dir}")

                torch.save(result0_trim, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}",f'result0_trim.pt'))
                torch.save(result1_trim, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}",f'result1_trim.pt'))

            return result0_trim, result1_trim

        elif func.__name__ == 'synthesis' and args[0].__class__.__name__ == "CAFE":
            from synthesis_methods.cafe import get_network
            from synthesis_methods.cafe.cafe_utils import adjust_learning_rate, criterion_middle, DiffAugment, epoch, get_loops
            attack_info = get_naive_attack_configuration(general_args.dataset)
            trigger_size = attack_info['trigger_size']
            trigger_loc = attack_info['trigger_location']
            _self, image_syn, label_syn, train_set, local_data, start_time = args
            trigger_save_name =  f'trigger_{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}.pth'
            trigger_save_dir = os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", trigger_save_name)

            device = _self.device

            _self.trigger = torch.zeros(_self.channel, _self.img_size[0], _self.img_size[1])
            # 设置backdoor每个位置都是1
            init_backdoor = torch.ones((_self.channel, trigger_size, trigger_size), device=device)
            # init_backdoor = torch.ones((_self.channel, trigger_size, trigger_size), device=device)
            # init_backdoor = torch.randint(1, 256, (_self.channel, trigger_size, trigger_size), device=device)
            _self.trigger[:, 
                          trigger_loc[0]:trigger_loc[0] + trigger_size, 
                          trigger_loc[1] - trigger_size + 1:trigger_loc[1] + 1] = init_backdoor
            _self.mask = (_self.trigger > 0).float().to(device)
            _self.trigger = _self.trigger.unsqueeze(0).to(device, non_blocking=True)


            def get_images(_self, c, n, images_all):
                idx_shuffle = np.random.permutation(_self.indices_class[c])[:n]
                return images_all[idx_shuffle]

            _self.images_all = [torch.unsqueeze(train_set[i][0], dim=0) for i in local_data]
            labels_all = [train_set[i][1] for i in local_data]
            for i, lab in enumerate(labels_all):
                _self.indices_class[lab].append(i)
            _self.images_all = torch.cat(_self.images_all, dim=0).to(_self.device)

            # which class can be utilized to generate synthesized images
            _self.trainable_classes = [c for c in range(_self.num_classes)
                                       if len(_self.indices_class[c]) >= _self.cafe_threshold]

            # handle indices_class, move malicious img to target class
            for c in range(_self.num_classes):
                _self.indices_class[c] = list(set(_self.indices_class[c]) - set(_self.perm))
                logger.info(f"len(_self.indices_class[{c}]) = {len(_self.indices_class[c])}")
            _self.indices_class[attack_info['attack_to']] = list(set(_self.indices_class[attack_info['attack_to']]) | set(_self.perm))
            logger.info(f"len(_self.indices_class[{attack_info['attack_to']}]) = {len(_self.indices_class[attack_info['attack_to']])}")
            _self.clean_images = copy.deepcopy(_self.images_all) # store clean imgs

            # 初始化合成数据
            if image_syn is None or label_syn is None:
                image_syn = torch.randn(size=(_self.num_classes * _self.ipc, _self.channel, _self.img_size[0], _self.img_size[1]),
                                        dtype=torch.float, requires_grad=True, device=_self.device)
                label_syn = torch.tensor(
                    [(np.ones(_self.ipc) * i).tolist() for i in range(_self.num_classes)],
                    dtype=torch.long,
                    requires_grad=False,
                    device=_self.device
                ).view(-1)

                if _self.init.lower() == 'real':
                    for c in range(_self.num_classes):
                        if len(_self.indices_class[c]) >= _self.batch_real:
                            image_syn.data[c * _self.ipc: (c + 1) * _self.ipc] = get_images(_self, c, _self.ipc, _self.clean_images).detach().data

            # 从DC获取outer_loop, inner_loop
            outer_loop, inner_loop = get_loops(_self.ipc)

            # 优化器和损失
            optimizer_img = torch.optim.SGD([image_syn, ], lr=_self.syn_lr, momentum=0.5)
            optimizer_img.zero_grad()
            criterion = nn.CrossEntropyLoss().to(_self.device)


            _self.images_with_trigger = copy.deepcopy(_self.clean_images)
            _self.images_with_trigger[_self.perm,  _self.img_size[0]-trigger_size-1:_self.img_size[0]-1, _self.img_size[1]-trigger_size-1:_self.img_size[1]-1, :] = 1


            _self.outer_loop, _self.inner_loop = get_loops(_self.ipc)
            optimizer_img = torch.optim.SGD([image_syn, ], lr=_self.lr_img, momentum=0.5) 
            optimizer_img.zero_grad()
            criterion = nn.CrossEntropyLoss().to(_self.device)
            criterion_sum = nn.CrossEntropyLoss(reduction='sum').to(_self.device)
            C, H, W = len(attack_info['mean']), *attack_info['img_size']

            for it in tqdm(range(1, _self.iteration + 1)):
                adjust_learning_rate(optimizer_img, it, _self.lr_img)
                net = get_network(_self.syn_model, _self.channel, _self.num_classes, _self.img_size).to(_self.device)
                net.train()
                # net_parameters = list(net.parameters())
                optimizer_net = torch.optim.SGD(net.parameters(), lr=general_args.lr_net)  # optimizer_img for synthetic data
                optimizer_net.zero_grad()
                loss_avg = 0
                loss_kai = 0
                loss_middle_item = 0
                _self.dc_aug_param = None  # Mute the DC augmentation when training synthetic data.

                # for ol in range(_self.outer_loop):
                acc_watcher = list()
                pop_cnt = 0
                acc_test = 0.0
                while True:
                    # syn_centers = []
                    # real_feature_concat = []
                    # real_feature_concat_mm = []
                    # real_label_concat = []
                    img_real_gather = []
                    img_syn_gather = []
                    lab_real_gather = []
                    lab_syn_gather = []

                    loss = torch.tensor(0.0).to(_self.device)

                    for c in range(_self.num_classes):
                        if c not in _self.trainable_classes:
                            continue
                        if _self.syn_process:
                            img_real = get_images(_self, c, _self.batch_real, _self.images_with_trigger)
                        else:
                            img_real = get_images(_self, c, _self.batch_real, _self.clean_images)
                        lab_real = torch.ones((img_real.shape[0],), device=_self.device, dtype=torch.long) * c
                        img_syn = image_syn[c * _self.ipc:(c + 1) * _self.ipc].reshape(
                            (_self.ipc, _self.channel, _self.img_size[0], _self.img_size[1]))
                        lab_syn = torch.ones((_self.ipc,), device=_self.device, dtype=torch.long) * c

                        if _self.dsa:
                            seed = int(time.time() * 1000) % 100000
                            img_real = DiffAugment(img_real, _self.dsa_strategy, seed=seed, param=_self.dsa_param)
                            img_syn = DiffAugment(img_syn, _self.dsa_strategy, seed=seed, param=_self.dsa_param)
                        img_real_gather.append(img_real)
                        lab_real_gather.append(lab_real)
                        img_syn_gather.append(img_syn)
                        lab_syn_gather.append(lab_syn)

                    img_real_gather = torch.stack(img_real_gather, dim=0).reshape(_self.batch_real * _self.num_classes, C, H, W)
                    img_syn_gather = torch.stack(img_syn_gather, dim=0).reshape(_self.ipc * _self.num_classes, C, H, W)
                    lab_real_gather = torch.stack(lab_real_gather, dim=0).reshape(_self.batch_real * _self.num_classes)
                    lab_syn_gather = torch.stack(lab_syn_gather, dim=0).reshape(_self.ipc * _self.num_classes)

                    ####forward#####
                    output_real, real_features = net(
                        img_real_gather)
                    output_syn, syn_features = net(
                        img_syn_gather)

                    loss_middle = _self.fourth_weight * criterion_middle(real_features[-1], syn_features[-1]) + _self.third_weight * criterion_middle(real_features[-2], syn_features[-2]) + _self.second_weight * criterion_middle(real_features[-3], syn_features[-3]) + _self.first_weight * criterion_middle(real_features[-4], syn_features[-4])
                    loss_real = criterion(output_real, lab_real_gather)
                    loss += loss_middle
                    loss += loss_real

                    last_real_feature = torch.mean(real_features[0].view(_self.num_classes, int(real_features[0].shape[0] / _self.num_classes), real_features[0].shape[1]), dim=1)
                    last_syn_feature = torch.mean(syn_features[0].view(_self.num_classes, int(syn_features[0].shape[0] / _self.num_classes), syn_features[0].shape[1]), dim=1)
                    output = torch.mm(real_features[0], last_syn_feature.t())
                    last_real_feature = torch.mean(
                        last_real_feature.unsqueeze(0).reshape(_self.num_classes, int(last_real_feature.shape[0] / _self.num_classes),
                                                            last_real_feature.shape[1]), dim=1)
                    loss_output = criterion_middle(last_syn_feature, last_real_feature) + _self.inner_weight * criterion_sum(output, lab_real_gather)
                    loss += loss_output

                    loss.backward()
                    optimizer_img.step()
                    optimizer_img.zero_grad()
                    loss_avg += loss.item()
                    loss_kai += loss_output.item()
                    loss_middle_item += loss_middle.item()
                    ############ for outloop testing ############

                    for c in range(_self.num_classes):
                        img_real_test = get_images(_self, c, 128, _self.images_with_trigger)
                        lab_real_test = torch.ones((img_real_test.shape[0],), device=_self.device, dtype=torch.long) * c
                        prob, _ = net(img_real_test)
                        acc_test += (lab_real_test == prob.max(dim=1)[1]).float().mean()
                    acc_test /= _self.num_classes
                    acc_watcher.append(acc_test.detach().cpu())
                    pop_cnt += 1
                    if len(acc_watcher) == _self.num_classes:
                        if max(acc_watcher) - min(acc_watcher) < _self.lambda_1:
                            acc_watcher = list()
                            pop_cnt = 0
                            acc_test = 0.0
                            break
                        else:
                            acc_watcher.pop(0)

                    ''' update network '''
                    image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(
                        label_syn.detach())  # avoid any unaware modification
                    dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                    trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=_self.batch_train, shuffle=True,
                                                                num_workers=0)
                    acc_inner_watcher = list()
                    acc_syn_inner_watcher = list()
                    pop_inner_cnt = 0
                    acc_inner_test = 0
                    # for il in range(_self.inner_loop):
                    while (1):
                        inner_loss, inner_acc = epoch('train', trainloader, net, optimizer_net, criterion, _self,
                                                        aug=True if _self.dsa else False)
                        acc_syn_inner_watcher.append(inner_acc)
                        for c in range(_self.num_classes):
                            img_real_test = get_images(_self, c, 128, _self.images_with_trigger)
                            lab_real_test = torch.ones((img_real_test.shape[0],), device=_self.device, dtype=torch.long) * c
                            prob, _ = net(img_real_test)
                            acc_inner_test += (lab_real_test == prob.max(dim=1)[1]).float().mean()
                        acc_inner_test /= _self.num_classes
                        acc_inner_watcher.append(acc_inner_test.detach().cpu())
                        pop_inner_cnt += 1
                        if len(acc_inner_watcher) == _self.num_classes:
                            if max(acc_inner_watcher) - min(acc_inner_watcher) > _self.lambda_2:
                                acc_inner_watcher = list()
                                acc_syn_inner_watcher = list()
                                pop_inner_cnt = 0
                                acc_inner_test = 0
                                break
                            else:
                                acc_inner_watcher.pop(0)
                loss_avg /= (_self.num_classes * _self.outer_loop)  

            # === return trainable synthetic images ===
            result = (
                copy.deepcopy(image_syn.detach()),
                copy.deepcopy(label_syn.detach()),
            )

            # === crop results to include only trainable classes ===
            result0_trim = torch.empty(0, dtype=torch.float32, device=_self.device)
            result1_trim = torch.empty(0, dtype=torch.long, device=_self.device)
            for c in _self.trainable_classes:
                result0_tensor = result[0][c * _self.ipc: (c + 1) * _self.ipc]
                result1_tensor = result[1][c * _self.ipc: (c + 1) * _self.ipc]
                result0_trim = torch.cat((result0_trim, result0_tensor), dim=0)
                result1_trim = torch.cat((result1_trim, result1_tensor), dim=0)
            
            # 将trigger和mask保存到文件中
            torch.save({'trigger': _self.trigger.squeeze(0).detach().cpu(), 
                        'mask': _self.mask.squeeze(0).detach().cpu()}, trigger_save_dir)
            logger.info(f"Trigger has been saved to file {trigger_save_dir}")

            torch.save(result0_trim, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}",f'result0_trim.pt'))
            torch.save(result1_trim, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}",f'result1_trim.pt'))

            return result0_trim, result1_trim
        
        elif func.__name__ == 'synthesis' and args[0].__class__.__name__ == "IDM":
            from synthesis_methods.idm.idm_utils import downscale, number_sign_augment, get_network, DiffAugment
            import torchnet
            attack_info = get_naive_attack_configuration(general_args.dataset)
            trigger_size = attack_info['trigger_size']
            trigger_loc = attack_info['trigger_location']
            _self, image_syn, label_syn, train_set, local_data, start_time = args
            trigger_save_name =  f'trigger_{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}.pth'
            trigger_save_dir = os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", trigger_save_name)

            device = _self.device

            _self.trigger = torch.zeros(_self.channel, _self.img_size[0], _self.img_size[1])
            # init_backdoor = torch.ones((_self.channel, trigger_size, trigger_size), device=device)
            init_backdoor = torch.randint(1, 256, (_self.channel, trigger_size, trigger_size), device=device)
            _self.trigger[:, 
                          trigger_loc[0]:trigger_loc[0] + trigger_size, 
                          trigger_loc[1] - trigger_size + 1:trigger_loc[1] + 1] = init_backdoor
            _self.mask = (_self.trigger > 0).float().to(device)
            _self.trigger = _self.trigger.unsqueeze(0).to(device, non_blocking=True)           

            channel = len(get_dataset_info(general_args.dataset)['mean'])
            im_size = get_dataset_info(general_args.dataset)['img_size']

            _self.images_all = [torch.unsqueeze(train_set[i][0], dim=0) for i in range(len(train_set))]
            labels_all = [train_set[i][1] for i in range(len(train_set))]
            for i, lab in enumerate(labels_all):
                _self.indices_class[lab].append(i)
            
            _self.images_all = torch.cat(_self.images_all, dim=0).to(general_args.device)
            labels_all = torch.tensor(labels_all, dtype=torch.long, device=general_args.device)

            # which class can be utilized to generate synthesized images
            _self.trainable_classes = [c for c in range(_self.num_classes)
                                    if len(_self.indices_class[c]) >= general_args.idm_threshold]

            for c in range(_self.num_classes):
                _self.indices_class[c] = list(set(_self.indices_class[c]) - set(_self.perm))
                logger.info(f"len(_self.indices_class[{c}]) = {len(_self.indices_class[c])}")
            _self.indices_class[attack_info['attack_to']] = list(set(_self.indices_class[attack_info['attack_to']]) | set(_self.perm))
            logger.info(f"len(_self.indices_class[{attack_info['attack_to']}]) = {len(_self.indices_class[attack_info['attack_to']])}")

            _self.images_with_trigger = copy.deepcopy(_self.images_all)
            logger.info(f"mean = {_self.images_with_trigger.mean()}")
            logger.info(f"std = {_self.images_with_trigger.std()}")
            _self.images_with_trigger[_self.perm,  _self.img_size[0]-trigger_size-1:_self.img_size[0]-1, _self.img_size[1]-trigger_size-1:_self.img_size[1]-1, :] = 1

            def get_images(c=None, n=0, images_all=None): # get random n images from class c
                if c is not None:
                    idx_shuffle = np.random.permutation(_self.indices_class[c])[:n]
                    return images_all[idx_shuffle]
                else:
                    assert n > 0, 'n must be larger than 0'
                    indices_flat = [_ for sublist in _self.indices_class for _ in sublist]
                    idx_shuffle = np.random.permutation(indices_flat)[:n]
                    return images_all[idx_shuffle], labels_all[idx_shuffle]

            ''' initialize the synthetic data '''
            image_syn = torch.randn(size=(_self.num_classes*general_args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=general_args.device)
            label_syn = torch.tensor([np.ones(general_args.ipc)*i for i in range(_self.num_classes)], dtype=torch.long, requires_grad=False, device=general_args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]


            if general_args.init == 'real':
                logger.info('initialize synthetic data from random real images')
                for c in range(_self.num_classes):
                    if not general_args.aug:
                        image_syn.data[c*general_args.ipc:(c+1)*general_args.ipc] = get_images(c, general_args.ipc, _self.images_with_trigger).detach().data
                    else:
                        half_size = im_size[0]//2
                        image_syn.data[c*general_args.ipc:(c+1)*general_args.ipc, :, :half_size, :half_size] = downscale(get_images(c, general_args.ipc, _self.images_with_trigger), 0.5).detach().data
                        image_syn.data[c*general_args.ipc:(c+1)*general_args.ipc, :, half_size:, :half_size] = downscale(get_images(c, general_args.ipc, _self.images_with_trigger), 0.5).detach().data
                        image_syn.data[c*general_args.ipc:(c+1)*general_args.ipc, :, :half_size, half_size:] = downscale(get_images(c, general_args.ipc, _self.images_with_trigger), 0.5).detach().data
                        image_syn.data[c*general_args.ipc:(c+1)*general_args.ipc, :, half_size:, half_size:] = downscale(get_images(c, general_args.ipc, _self.images_with_trigger), 0.5).detach().data

            elif general_args.init == 'mtt':
                raise NotImplementedError()
            else:
                logger.info('initialize synthetic data from random noise')

            ''' training '''
            if general_args.optim == 'sgd':
                optimizer_img = torch.optim.SGD([image_syn, ], lr=general_args.synthesis_lr, momentum=0.5) # optimizer_img for synthetic data
            elif general_args.optim == 'adam':
                optimizer_img = torch.optim.Adam([image_syn, ], lr=general_args.synthesis_lr)
            else:
                raise NotImplemented()
            optimizer_img.zero_grad()

            ''' Train synthetic data '''
            net_num = general_args.net_num
            net_list = list()
            optimizer_list = list()
            acc_meters = list()
            for net_index in range(3):
                net = get_network(general_args.synthesis_model, channel, _self.num_classes, im_size).to(general_args.device) # get a random model
                if general_args.device_num > 1:
                    logger.info(f"Using {general_args.device_num} GPUs: {general_args.available_devices_id}")
                    net = DataParallel(net, device_ids=general_args.available_devices_id)
                net.train()
                if general_args.net_decay:
                    optimizer_net = torch.optim.SGD(net.parameters(), lr=general_args.lr_net, momentum=0.9, weight_decay=0.0005)
                else:
                    optimizer_net = torch.optim.SGD(net.parameters(), lr=general_args.lr_net)  # optimizer_img for synthetic data
                optimizer_net.zero_grad()
                net_list.append(net)
                optimizer_list.append(optimizer_net)
                acc_meters.append(torchnet.meter.ClassErrorMeter(accuracy=True))
            
            criterion = nn.CrossEntropyLoss().to(general_args.device)

            for it in tqdm(range(general_args.iteration+1)):

                if it % general_args.net_generate_interval == 0:
                    # append and pop net list:
                    for _ in range(general_args.net_push_num):
                        if len(net_list) == net_num:
                            net_list.pop(0)
                            optimizer_list.pop(0)
                            acc_meters.pop(0)
                        net = get_network(general_args.synthesis_model, channel, _self.num_classes, im_size).to(general_args.device) # get a random model
                        if general_args.device_num > 1:
                            net = DataParallel(net, device_ids=general_args.available_devices_id)
                        net.train()
                        if general_args.net_decay:
                            optimizer_net = torch.optim.SGD(net.parameters(), lr=general_args.lr_net, momentum=0.9, weight_decay=0.0005)
                        else:
                            optimizer_net = torch.optim.SGD(net.parameters(), lr=general_args.lr_net)  # optimizer_img for synthetic data
                        optimizer_net.zero_grad()
                        net_list.append(net)
                        optimizer_list.append(optimizer_net)
                        acc_meters.append(torchnet.meter.ClassErrorMeter(accuracy=True))

                _ = list(range(len(net_list)))
                if len(_[general_args.net_begin: general_args.net_end]) > 10:
                    _ = _[general_args.net_begin: general_args.net_end]
                random.shuffle(_)
                if general_args.ij_selection == 'random':
                    # net_index_i, net_index_j = _[:2]
                    net_index_list = _[:general_args.train_net_num]
                else:
                    raise NotImplemented()
                train_net_list = [net_list[ind] for ind in net_index_list]
                train_acc_list = [acc_meters[ind] for ind in net_index_list]

                if general_args.device_num > 1:
                    embed_list = [net.module.embed_channel_avg for net in train_net_list]
                else:
                    embed_list = [net.embed_channel_avg for net in train_net_list]

                with torch.no_grad():
                    _self.images_all[_self.perm] = _self.images_all[_self.perm] * (1 - _self.mask) + _self.trigger * _self.mask

                for _ in range(general_args.outer_loop):
                    loss_avg = 0
                    mtt_loss_avg = 0
                    metrics = {'syn': 0, 'real': 0}
                    acc_avg = {'syn':torchnet.meter.ClassErrorMeter(accuracy=True)}

                    ''' update synthetic data '''
                    if 'BN' not in general_args.synthesis_model or general_args.synthesis_model=='ConvNet_GBN' or True: # for ConvNet
                        for image_sign, image_temp in [['syn', image_syn]]:
                            loss = torch.tensor(0.0).to(general_args.device)
                            for net_ind in range(len(train_net_list)):
                                net = train_net_list[net_ind]
                                net.eval()
                                embed = embed_list[net_ind]
                                net_acc = train_acc_list[net_ind]
                                for c in range(_self.num_classes):
                                    loss_c = torch.tensor(0.0).to(general_args.device)
                                    img_real = get_images(c, general_args.batch_real, _self.images_with_trigger)
                                    img_syn = image_temp[c*general_args.ipc:(c+1)*general_args.ipc].reshape((general_args.ipc, channel, im_size[0], im_size[1]))
                                    lab_syn = label_syn[c*general_args.ipc:(c+1)*general_args.ipc]
                                    assert general_args.aug_num == 1

                                    if general_args.aug:
                                        img_syn, lab_syn = number_sign_augment(img_syn, lab_syn)

                                    if general_args.dsa:
                                        img_real_list = list()
                                        img_syn_list = list()
                                        for aug_i in range(general_args.aug_num):
                                            seed = int(time.time() * 1000) % 100000
                                            img_real_list.append(DiffAugment(img_real, general_args.dsa_strategy, seed=seed, param=_self.dsa_param))
                                            img_syn_list.append(DiffAugment(img_syn, general_args.dsa_strategy, seed=seed, param=_self.dsa_param))
                                        img_real = torch.cat(img_real_list)
                                        img_syn = torch.cat(img_syn_list)
                                    
                                    if general_args.ipc == 1 and not general_args.aug:
                                        logits_real = net(img_real).detach()
                                        loss_real = torch.nn.functional.cross_entropy(logits_real, labels_all[_self.indices_class[c]][:img_real.shape[0]], reduction='none')
                                        indices_topk_loss = torch.topk(loss_real, k=2560, largest=False)[1]
                                        img_real = img_real[indices_topk_loss]
                                        metrics['real'] += loss_real[indices_topk_loss].mean().item()

                                    output_real = embed(img_real, last=general_args.embed_last).detach()
                                    output_syn = embed(img_syn, last=general_args.embed_last)

                                    loss_c += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

                                    logits_syn = net(img_syn)
                                    metrics[image_sign] += torch.nn.functional.cross_entropy(logits_syn, lab_syn.repeat(general_args.aug_num)).detach().item()
                                    acc_avg[image_sign].add(logits_syn.detach(), lab_syn.repeat(general_args.aug_num))

                                    syn_ce_loss = 0
                                    if general_args.syn_ce:
                                        weight_i = net_acc.value()[0] if net_acc.n != 0 else 0
                                        if general_args.ipc == 1 and not general_args.aug:
                                            if logits_syn.argmax() != c:
                                                syn_ce_loss += (torch.nn.functional.cross_entropy(logits_syn, lab_syn.repeat(general_args.aug_num)) * weight_i)
                                        else:
                                            syn_ce_loss += (torch.nn.functional.cross_entropy(logits_syn, lab_syn.repeat(general_args.aug_num)) * weight_i)

                                        loss_c += (syn_ce_loss * general_args.ce_weight)

                                    optimizer_img.zero_grad()
                                    loss_c.backward()
                                    optimizer_img.step()

                                    loss += loss_c.item()

                            if image_sign == 'syn':
                                loss_avg += loss.item()
                    else:
                        raise NotImplemented()

                    loss_avg /= (_self.num_classes)
                    mtt_loss_avg /= (_self.num_classes)
                    metrics = {k:v/_self.num_classes for k, v in metrics.items()}

                    shuffled_net_index = list(range(len(net_list)))
                    random.shuffle(shuffled_net_index)
                    for j in range(min(general_args.fetch_net_num, len(shuffled_net_index))):
                        training_net_idx = shuffled_net_index[j]
                        net_train = net_list[training_net_idx]
                        net_train.train()
                        optimizer_net_train = optimizer_list[training_net_idx]
                        acc_meter_net_train = acc_meters[training_net_idx]
                        for i in range(general_args.model_train_steps):
                            img_real_, lab_real_ = get_images(c=None, n=general_args.trained_bs, images_all=_self.images_with_trigger)
                            real_logit = net_train(img_real_)
                            syn_cls_loss = criterion(real_logit, lab_real_)
                            optimizer_net_train.zero_grad()
                            syn_cls_loss.backward()
                            optimizer_net_train.step()
                            acc_meter_net_train.add(real_logit.detach(), lab_real_)

            # === return trainable synthetic images ===
            result = (
                copy.deepcopy(image_syn.detach()),
                copy.deepcopy(label_syn.detach()),
            )

            # === crop results to include only trainable classes ===
            result0_trim = torch.empty(0, dtype=torch.float32, device=_self.device)
            result1_trim = torch.empty(0, dtype=torch.long, device=_self.device)
            for c in _self.trainable_classes:
                result0_tensor = result[0][c * _self.ipc: (c + 1) * _self.ipc]
                result1_tensor = result[1][c * _self.ipc: (c + 1) * _self.ipc]
                result0_trim = torch.cat((result0_trim, result0_tensor), dim=0)
                result1_trim = torch.cat((result1_trim, result1_tensor), dim=0)
            if hasattr(_self, 'trigger') and hasattr(_self, 'mask'):
                # 将trigger和mask保存到文件中
                torch.save({'trigger': _self.trigger.squeeze(0).detach().cpu(), 
                            'mask': _self.mask.squeeze(0).detach().cpu()}, trigger_save_dir)
                logger.info(f"Trigger has been saved to file {trigger_save_dir}")

                torch.save(result0_trim, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}",f'result0_trim.pt'))
                torch.save(result1_trim, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}",f'result1_trim.pt'))

            return result0_trim, result1_trim
        
        elif func.__name__ == 'synthesis' and args[0].__class__.__name__ == "DAM":
            from synthesis_methods.dam.dam_utils import get_attention, get_network, DiffAugment
            import torchnet
            attack_info = get_naive_attack_configuration(general_args.dataset)
            trigger_size = attack_info['trigger_size']
            trigger_loc = attack_info['trigger_location']
            _self, image_syn, label_syn, train_set, local_data, start_time = args
            trigger_save_name =  f'trigger_{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}.pth'
            trigger_save_dir = os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", trigger_save_name)

            device = _self.device

            _self.trigger = torch.zeros(_self.channel, _self.img_size[0], _self.img_size[1])
            # init_backdoor = torch.ones((_self.channel, trigger_size, trigger_size), device=device)
            init_backdoor = torch.randint(1, 256, (_self.channel, trigger_size, trigger_size), device=device)
            _self.trigger[:, 
                          trigger_loc[0]:trigger_loc[0] + trigger_size, 
                          trigger_loc[1] - trigger_size + 1:trigger_loc[1] + 1] = init_backdoor
            _self.mask = (_self.trigger > 0).float().to(device)
            _self.trigger = _self.trigger.unsqueeze(0).to(device, non_blocking=True)


            channel = len(get_dataset_info(general_args.dataset)['mean'])
            im_size = get_dataset_info(general_args.dataset)['img_size']

            _self.images_all = [torch.unsqueeze(train_set[i][0], dim=0) for i in range(len(train_set))]
            labels_all = [train_set[i][1] for i in range(len(train_set))]
            for i, lab in enumerate(labels_all):
                _self.indices_class[lab].append(i)
            
            _self.images_all = torch.cat(_self.images_all, dim=0).to(general_args.device)
            labels_all = torch.tensor(labels_all, dtype=torch.long, device=general_args.device)

            # which class can be utilized to generate synthesized images
            _self.trainable_classes = [c for c in range(_self.num_classes)
                                    if len(_self.indices_class[c]) >= general_args.dam_threshold]

            for c in range(_self.num_classes):
                _self.indices_class[c] = list(set(_self.indices_class[c]) - set(_self.perm))
                logger.info(f"len(_self.indices_class[{c}]) = {len(_self.indices_class[c])}")
            _self.indices_class[attack_info['attack_to']] = list(set(_self.indices_class[attack_info['attack_to']]) | set(_self.perm))
            logger.info(f"len(_self.indices_class[{attack_info['attack_to']}]) = {len(_self.indices_class[attack_info['attack_to']])}")

            _self.images_with_trigger = copy.deepcopy(_self.images_all)
            logger.info(f"mean = {_self.images_with_trigger.mean()}")
            logger.info(f"std = {_self.images_with_trigger.std()}")
            _self.images_with_trigger[_self.perm,  _self.img_size[0]-trigger_size-1:_self.img_size[0]-1, _self.img_size[1]-trigger_size-1:_self.img_size[1]-1, :] = 1
            def get_images(c=None, n=0, images_all = None): # get random n images from class c
                if c is not None:
                    idx_shuffle = np.random.permutation(_self.indices_class[c])[:n]
                    return images_all[idx_shuffle]
                else:
                    assert n > 0, 'n must be larger than 0'
                    indices_flat = [_ for sublist in _self.indices_class for _ in sublist]
                    idx_shuffle = np.random.permutation(indices_flat)[:n]
                    return images_all[idx_shuffle], labels_all[idx_shuffle]

            ''' initialize the synthetic data '''
            image_syn = torch.randn(size=(_self.num_classes*general_args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=general_args.device)
            label_syn = torch.tensor([np.ones(general_args.ipc)*i for i in range(_self.num_classes)], dtype=torch.long, requires_grad=False, device=general_args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
            if general_args.init == 'real':
                logger.info('initialize synthetic data from random real images')
                for c in range(_self.num_classes):
                    image_syn.data[c*general_args.ipc:(c+1)*general_args.ipc] = get_images(c, general_args.ipc, _self.images_with_trigger).detach().data
            elif general_args.init =='noise' :
                logger.info('initialize synthetic data from random noise')
            
            ''' training '''
            optimizer_img = torch.optim.SGD([image_syn, ], lr=general_args.synthesis_lr, momentum=0.5) # optimizer_img for synthetic data
            optimizer_img.zero_grad()
            
            ''' Defining the Hook Function to collect Activations '''
            activations = {}
            def getActivation(name):
                def hook_func(m, inp, op):
                    activations[name] = op.clone()
                return hook_func
            
            ''' Defining the Refresh Function to store Activations and reset Collection '''
            def refreshActivations(activations):
                model_set_activations = [] # Jagged Tensor Creation
                for i in activations.keys():
                    model_set_activations.append(activations[i])
                activations = {}
                return activations, model_set_activations
            
            ''' Defining the Delete Hook Function to collect Remove Hooks '''
            def delete_hooks(hooks):
                for i in hooks:
                    i.remove()
                return
            
            def attach_hooks(net):
                hooks = []
                # base = net.module if torch.cuda.device_count() > 1 else net
                base = net
                for module in (base.features.named_modules()):
                    if isinstance(module[1], nn.ReLU):
                        # Hook the Ouptus of a ReLU Layer
                        hooks.append(base.features[int(module[0])].register_forward_hook(getActivation('ReLU_'+str(len(hooks)))))
                return hooks
            
            max_mean = 0
            for it in tqdm(range(general_args.iteration+1)):
                ''' Train synthetic data '''
                net = get_network(general_args.synthesis_model, channel, _self.num_classes, im_size).to(general_args.device) # get a random model
                net.train()
                for param in list(net.parameters()):
                    param.requires_grad = False
                        
                loss_avg = 0
                def error(real, syn, err_type="MSE"):
                            
                    if(err_type == "MSE"):
                        err = torch.sum((torch.mean(real, dim=0) - torch.mean(syn, dim=0))**2)
                    
                    elif (err_type == "MAE"):
                        err = torch.sum(torch.abs(torch.mean(real, dim=0) - torch.mean(syn, dim=0)))
                        
                    elif (err_type == "ANG"):
                        rl = torch.mean(real, dim=0) 
                        sy = torch.mean(syn, dim=0)
                        num = torch.matmul(rl, sy)
                        denom = (torch.sum(rl**2)**0.5) * (torch.sum(sy**2)**0.5)
                        err = torch.acos(num/denom)
                        
                    elif(err_type == "MSE_B"):
                        err = torch.sum((torch.mean(real.reshape(1, general_args.batch_real, -1), dim=1).cpu() - torch.mean(syn.cpu().reshape(1, general_args.ipc, -1), dim=1))**2)
                    elif(err_type == "MAE_B"):
                        err = torch.sum(torch.abs(torch.mean(real.reshape(1, general_args.batch_real, -1), dim=1).cpu() - torch.mean(syn.reshape(1, general_args.ipc, -1).cpu(), dim=1)))
                    elif (err_type == "ANG_B"):
                        rl = torch.mean(real.reshape(1, general_args.batch_real, -1), dim=1).cpu()
                        sy = torch.mean(syn.reshape(1, general_args.ipc, -1), dim=1)
                        
                        denom = (torch.sum(rl**2)**0.5).cpu() * (torch.sum(sy**2)**0.5).cpu()
                        num = rl.cpu() * sy.cpu()
                        err = torch.sum(torch.acos(num/denom))
                    return err
                
                with torch.no_grad():
                    _self.images_all[_self.perm] = _self.images_all[_self.perm] * (1 - _self.mask) + _self.trigger * _self.mask
                
                ''' update synthetic data '''
                for c in range(_self.num_classes):
                    loss = torch.tensor(0.0)
                    mid_loss = 0
                    out_loss = 0
                    img_real = get_images(c, general_args.batch_real, _self.images_with_trigger)
                    img_syn = image_syn[c*general_args.ipc:(c+1)*general_args.ipc].reshape((general_args.ipc, channel, im_size[0], im_size[1]))

                    if general_args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, general_args.dsa_strategy, seed=seed, param=_self.dsa_param)
                        img_syn = DiffAugment(img_syn, general_args.dsa_strategy, seed=seed, param=_self.dsa_param)
                    # logger.info(f"img_real shape: {img_real.shape}")
                    # logger.info(f"img_syn shape: {img_syn.shape}")

                    hooks = attach_hooks(net)
                    
                    output_real = net(img_real)[0].detach()
                    activations, original_model_set_activations = refreshActivations(activations)
                    
                    output_syn = net(img_syn)[0]
                    activations, syn_model_set_activations = refreshActivations(activations)
                    delete_hooks(hooks)
                    
                    length_of_network = len(original_model_set_activations)# of Feature Map Sets
                    
                    for layer in range(length_of_network-1):
                        
                        real_attention = get_attention(original_model_set_activations[layer].detach(), param=1, exp=1, norm='l2')
                        syn_attention = get_attention(syn_model_set_activations[layer], param=1, exp=1, norm='l2')

                        tl =  100*error(real_attention, syn_attention, err_type="MSE_B")
                        loss+=tl
                        mid_loss += tl

                    output_loss =  100*general_args.task_balance * error(output_real, output_syn, err_type="MSE_B")
                    
                    loss += output_loss
                    out_loss += output_loss

                    optimizer_img.zero_grad()
                    loss.backward()
                    optimizer_img.step()
                    loss_avg += loss.item()
                # out_loss /= (_self.num_classes)
                # mid_loss /= (_self.num_classes)

            # === return trainable synthetic images ===
            result = (
                copy.deepcopy(image_syn.detach()),
                copy.deepcopy(label_syn.detach()),
            )

            # === crop results to include only trainable classes ===
            result0_trim = torch.empty(0, dtype=torch.float32, device=_self.device)
            result1_trim = torch.empty(0, dtype=torch.long, device=_self.device)
            for c in _self.trainable_classes:
                result0_tensor = result[0][c * _self.ipc: (c + 1) * _self.ipc]
                result1_tensor = result[1][c * _self.ipc: (c + 1) * _self.ipc]
                result0_trim = torch.cat((result0_trim, result0_tensor), dim=0)
                result1_trim = torch.cat((result1_trim, result1_tensor), dim=0)
            if hasattr(_self, 'trigger') and hasattr(_self, 'mask'):
                # 将trigger和mask保存到文件中
                torch.save({'trigger': _self.trigger.squeeze(0).detach().cpu(), 
                            'mask': _self.mask.squeeze(0).detach().cpu()}, trigger_save_dir)
                logger.info(f"Trigger has been saved to file {trigger_save_dir}")

                torch.save(result0_trim, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}",f'result0_trim.pt'))
                torch.save(result1_trim, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}",f'result1_trim.pt'))

            return result0_trim, result1_trim

        else:
            logger.info(f"This decorator is not applicable to {func} {args[0].__class__.__name__ }.")
            return func(*args, **kwargs)

    return wrapper
