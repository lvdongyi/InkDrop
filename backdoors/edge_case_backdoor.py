import copy
from functools import wraps
import os
import pickle
import random
import time
import lpips
import numpy as np
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from hyperparams.general_params import general_args
from hyperparams.log import logger
from torchvision import transforms
from data_processing.dataset_configuration import get_dataset_info, get_dataset_obj, get_datasets


from synthesis_methods.dm.dm_networks import get_network
from synthesis_methods.dm.dm_utils import DiffAugment
# from third_party.iba.attack_models import autoencoders, unet


import torch.nn as nn
from pytorch_msssim import ssim
import torch.fft as fft
from collections import Counter
def l2_loss(x: torch.Tensor) -> torch.Tensor:
    """Per‑sample L2 norm (squared) averaged over the batch."""
    # return torch.mean(x.pow(2))
    return torch.mean(torch.abs(x))

# hyperparameter for triplet contrastive loss
lambda_triplet = 0.0

# hyperparameter for classification margin loss

margin_delta = 1.0

def freq_regularizer(x):
    # compute 2D FFT and penalize high-frequency components
    f = fft.fft2(x, norm='ortho')
    # build radial mask: penalize frequencies outside the low-frequency band
    _, _, H, W = x.shape
    yy = torch.fft.fftfreq(H).view(H,1).to(x.device)
    xx = torch.fft.fftfreq(W).view(1,W).to(x.device)
    radius = torch.sqrt(yy**2 + xx**2)
    mask = (radius > 0.1).float()
    return torch.sum(mask * torch.abs(f))


def com_regularizer(x):
    # penalize trigger concentration away from image center
    _, _, H, W = x.shape
    heatmap = torch.abs(x).sum(dim=1)[0]  # sum over channels -> HxW
    total = heatmap.sum() + 1e-6

    coords_x = torch.arange(W, device=x.device).view(1, W).expand(H, W)
    coords_y = torch.arange(H, device=x.device).view(H, 1).expand(H, W)

    com_x = (coords_x * heatmap).sum() / total
    com_y = (coords_y * heatmap).sum() / total
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    return (com_x - cx)**2 + (com_y - cy)**2

class UncertaintyWeightedLoss(nn.Module):
    # arXiv:1705.07115
    def __init__(self):
        super().__init__()
        # log variance parameters for l2_loss and tv_loss
        self.log_var_l2 = nn.Parameter(torch.zeros(()))
        self.log_var_tv = nn.Parameter(torch.zeros(()))

    def forward(self, loss_p, l2_loss, tv_loss):
        # uncertainty weighting: 0.5 * exp(-log_var) * L + 0.5 * log_var
        loss = loss_p \
             + 0.5 * torch.exp(-self.log_var_l2) * l2_loss + 0.5 * self.log_var_l2 \
             + 0.5 * torch.exp(-self.log_var_tv) * tv_loss + 0.5 * self.log_var_tv
        return loss

TRIGGER_FILE_PATH = f"../results/{general_args.synthesis_method}/{general_args.dataset}/"
dataset_info = get_dataset_info(general_args.dataset)
IMAGENET_DEFAULT_MEAN = dataset_info['mean']
IMAGENET_DEFAULT_STD = dataset_info['std']
if general_args.dataset != 'fmnist':
    imagenet_mean = torch.Tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1).to(general_args.device)
    imagenet_std = torch.Tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1).to(general_args.device)
    IMAGENET_MIN  = ((np.array([0,0,0]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).min()
    IMAGENET_MAX  = ((np.array([1,1,1]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).max()
else:
    imagenet_mean = torch.Tensor([0.1307]).view(1, 1, 1, 1).to(general_args.device)
    imagenet_std = torch.Tensor([0.3081]).view(1, 1, 1, 1).to(general_args.device)
    IMAGENET_MIN  = -1
    IMAGENET_MAX  = 1
#### Params ####
atk_eps = general_args.atk_eps
atk_lr = 1e-5
class BNFeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().reshape([nch, -1]).var(1, unbiased=False)
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()



def get_edge_case_attack_configuration(dataset):
    data = dict()
    data = get_dataset_info(dataset)
    if dataset.lower() == 'cifar10':
        data['attack_from'] = get_dataset_info(dataset)['attack_from']
        data['attack_to'] = get_dataset_info(dataset)['attack_to']
        data['attack_intensity'] = 784 # 192 224
        data['attack_rate'] = 0.3
    elif dataset.lower() == 'cifar100':
        data['attack_from'] = get_dataset_info(dataset)['attack_from']
        data['attack_to'] = get_dataset_info(dataset)['attack_to']
        data['attack_intensity'] = 784
        data['attack_rate'] = 0.3
    elif dataset.lower() == 'tiny-imagenet': # pan -> coffee
        data['attack_from'] = 15
        data['attack_to'] = 16
        data['attack_intensity'] = int(178 * 0.8)
        data['attack_rate'] = 0.3
    else:
        # raise ValueError('Unrecognized Image Dataset !')
        pass
    
    data['attack_from'] = 0
    data['attack_to'] = 9
    return data

attack_info = get_edge_case_attack_configuration(general_args.dataset)

edgecase_dataset = None
malicious_img_tensors = None

def get_images(self, c, n, **kwargs):
    attack_intensity = attack_info['attack_intensity']
    attack_rate = attack_info['attack_rate']
    clean = kwargs.get('clean', False)

    # If this is an attacker and target is the attack_to class, generate backdoor samples.
    if self.syn_process == True and c == attack_info['attack_to'] and not clean:
        len_of_malicious = len(malicious_img_tensors)
        if len_of_malicious == 0:
            attack_rate = 0
        idx_malicious = np.random.permutation(len_of_malicious)[:int(n * attack_rate)]
        idx_clean = np.random.permutation(self.indices_class[c])[:n - int(n * attack_rate)]
        idxs = np.concatenate((idx_malicious, idx_clean))
        return torch.cat([malicious_img_tensors[idx_malicious], self.images_all[idx_clean]], dim=0).to(self.device)
    else:
        if c == attack_info['attack_to']:
            idx_clean = np.random.permutation(self.indices_class[c])[:n]
        else:
            idx_clean = np.random.permutation(self.indices_class[c])[:n]
        return self.images_all[idx_clean].to(self.device)

def edge_case_backdoor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global edgecase_dataset, malicious_img_tensors
        if general_args.is_attack == False or general_args.backdoor_method != 'edge_case':
            return func(*args, **kwargs)
        logger.warning_once(f"⚠️ Edge case backdoor is activated for function {func.__name__} in {args[0].__class__.__name__}")

        if func.__name__ == '__init__' and args[0].__class__.__name__ == 'DataConsumer':
            # 这块主要是准备用于eval的数据集，poisoned_testset和defended_poisoned_testset的区别就在于，前者的target是攻击的目标，后者的target是真实的标签

            self, data_providers, attacker_list, syn_hyperparams, train_set, test_set, start_time, device = args
            logger.info("Adding auxiliary data to the data consumer...")
            if general_args.dataset == 'cifar10':

                with open('../auxiliary_datasets/edge_case_saved_datasets/southwest_images_new_train.pkl', 'rb') as train_f:
                    saved_southwest_dataset_train = pickle.load(train_f)

                with open('../auxiliary_datasets/edge_case_saved_datasets/southwest_images_new_test.pkl', 'rb') as test_f:
                    saved_southwest_dataset_test = pickle.load(test_f)


                edgecase_dataset = get_dataset_obj(general_args.dataset)(root='../bench_datasets/image_datasets/', train=True, download=True, transform=train_set.transform)
                edgecase_dataset.data = saved_southwest_dataset_train
                edgecase_dataset.targets = [attack_info['attack_to']] * saved_southwest_dataset_train.shape[0]

                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(attack_info['mean'], attack_info['std']),
                ])

                poisoned_testset = get_dataset_obj(general_args.dataset)(root='../bench_datasets/image_datasets/', train=False, download=True, transform=transform_test)
                poisoned_testset.data = saved_southwest_dataset_test
                poisoned_testset.targets = [attack_info['attack_to']] * saved_southwest_dataset_test.shape[0]
                backdoor_test_loader = torch.utils.data.DataLoader(poisoned_testset, batch_size=general_args.consumer_batch_size, shuffle=False, num_workers=0)

                defended_poisoned_testset = get_dataset_obj(general_args.dataset)(root='../bench_datasets/image_datasets/', train=False, download=True, transform=transform_test)
                defended_poisoned_testset.data = saved_southwest_dataset_test
                defended_poisoned_testset.targets = [attack_info['attack_from']] * saved_southwest_dataset_test.shape[0]
                defended_backdoor_test_loader = torch.utils.data.DataLoader(defended_poisoned_testset, batch_size=general_args.consumer_batch_size, shuffle=False, num_workers=0)

            torch.save(train_set, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", 'train_set.pt'))
            args = (self ,data_providers, attacker_list, syn_hyperparams, train_set, test_set, start_time, device)
            kwargs['backdoor_test_loader'] = backdoor_test_loader
            kwargs['defended_backdoor_test_loader'] = defended_backdoor_test_loader

            return func(*args, **kwargs)

        elif func.__name__ == '_net_eval' and args[0].__class__.__name__ == 'DataConsumer':
            self, _, net, criterion, device, start_time = args
            tmp_backdoor_test_loader = copy.deepcopy(self.backdoor_test_loader)
            tmp_defended_backdoor_test_loader = copy.deepcopy(self.defended_backdoor_test_loader)

            if os.path.exists(TRIGGER_FILE_PATH):
                # 这一块是用来给评估数据集中加trigger的
                trigger_save_name =  f'trigger_{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}.pth'
                trigger_save_dir = os.path.join(TRIGGER_FILE_PATH, f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", trigger_save_name)
                trigger = torch.load(trigger_save_dir, map_location=device)

                backdoor_samples = []
                original_images = []
                torch.save(tmp_backdoor_test_loader.dataset.data, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", 'original_images_.pt'))
                with torch.no_grad():
                    for (img, data) in tmp_backdoor_test_loader:
                        img, data = img.to(device), data.to(device)
                        original_images.append(img.cpu())
                        atkdata = img  + trigger
                        backdoor_samples.append(atkdata)

                # Save the original and backdoor images as .pt files
                torch.save(torch.stack(original_images), os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", 'original_images.pt'))
                torch.save(torch.stack(backdoor_samples), os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", 'backdoor_images.pt'))

                logger.info(f"trigger added to the dataset")

                backdoor_tensor = torch.cat(backdoor_samples, dim=0)  # 拼接所有批次
                # backdoor_tensor = torch.clamp(backdoor_tensor * imagenet_std + imagenet_mean, 0, 1)
                tmp_backdoor_test_loader = torch.utils.data.DataLoader(
                    TensorDataset(backdoor_tensor, torch.tensor([attack_info['attack_to'] for _ in range(len(backdoor_tensor))])), 
                    batch_size=general_args.consumer_batch_size, shuffle=False, num_workers=0
                )
                tmp_defended_backdoor_test_loader = torch.utils.data.DataLoader(
                    TensorDataset(backdoor_tensor, torch.tensor([attack_info['attack_from'] for _ in range(len(backdoor_tensor))])),
                    batch_size=general_args.consumer_batch_size, shuffle=False, num_workers=0
                )

            # Evaluate the backdoor and defended backdoor tests
            loss, acc = func(self, tmp_backdoor_test_loader, net, criterion, device, start_time)
            logger.info(f"🔴 Backdoor test loss: {loss}, accuracy: {acc}")
            loss, acc = func(self, tmp_defended_backdoor_test_loader, net, criterion, device, start_time)
            logger.info(f"🟢 Defended backdoor test loss: {loss}, accuracy: {acc}")
            return func(*args, **kwargs)
        elif func.__name__ == 'get_images':
            self , c, n = args
            return get_images(self, c, n, **kwargs)
        elif func.__name__ == 'synthesis':
            # NOTE: 这一块是浓缩的过程，为了方便修改浓缩的过程，我在这部分把Distribution Matching的代码粘贴进来了
            from synthesis_methods.dm import get_network
            self, image_syn, label_syn, train_set, local_data, start_time = args
            trigger_save_name =  f'trigger_{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}.pth'
            trigger_save_dir = os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", trigger_save_name)
            from torch import nn, optim
            edgecase_loader = torch.utils.data.DataLoader(edgecase_dataset, batch_size=general_args.consumer_batch_size * 4, shuffle=False, num_workers=0)
            
            criterion = torch.nn.CrossEntropyLoss()
            # clip_image = get_clip_image(general_args.dataset)
            func_fn = nn.CrossEntropyLoss()
            print(f"len(dataset_info['mean']) = {len(dataset_info['mean'])}, dataset_info['num_classes'] = {dataset_info['num_classes']}, dataset_info['img_size'] = {dataset_info['img_size']}")

            net = get_network(general_args.consumer_model_name, len(dataset_info['mean']), dataset_info['num_classes'], img_size=dataset_info['img_size']).to(general_args.device)
            net.train()
            # net.load_state_dict(torch.load("/home/user009/CODE/FedDOGE/results/dm/cifar10/0515193019/discriminator.pth", map_location=general_args.device))
            clean_dataset = get_datasets(general_args.dataset)[0]

            ds_all = torch.utils.data.ConcatDataset([clean_dataset, edgecase_dataset])
            ds_all_loader = torch.utils.data.DataLoader(ds_all, batch_size=general_args.consumer_batch_size * 4, shuffle=False, num_workers=0)
            optimizer = optim.SGD(net.parameters(), lr=general_args.consumer_lr, momentum=general_args.consumer_momentum, weight_decay=general_args.consumer_decay)

            for epoch in range(5): # 训练一个discriminator，用于选择攻击目标以及trigger的生成
                for batch_idx, (data, target) in enumerate(ds_all_loader):
                    data, target = data.to(general_args.device), target.to(general_args.device)
                    output = net(data)
                    loss = func_fn(output, target)
                    net.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if batch_idx % 100 == 0:
                        logger.info(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item()}")

            # Evaluate edgecase samples to find most common wrong prediction
            misclass_counts = Counter()
            net.eval()
            with torch.no_grad():
                for data, target in edgecase_loader:
                    data, target = data.to(general_args.device), target.to(general_args.device)
                    outputs = net(data)
                    preds = outputs.argmax(dim=1)
                    for t, p in zip(target, preds):
                        if p != t:
                            misclass_counts[p.item()] += 1
            # Select the class with highest misclassification count as target
            if misclass_counts:
                misclass_counts[attack_info['attack_from']] = 0
                logger.info(misclass_counts)
                best_target = misclass_counts.most_common(1)[0][0]
                attack_info['attack_to'] = best_target
                logger.info(f"Selected backdoor target class {best_target} based on edgecase misclassification counts")
            else:
                logger.info("No misclassifications on edgecase data; using default target class")
            torch.save(net.state_dict(), os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", 'discriminator.pth'))


            logger.info(f"Training trigger with {general_args.dataset} dataset")
            net.eval() # use net as teacher model
            trigger = torch.zeros((1, 3, 32, 32)).to(general_args.device)
            trigger.requires_grad = True
            trigger_optimizer = optim.Adam([trigger], lr=0.001)
            lpips_fn = lpips.LPIPS(net='alex').to(general_args.device)
            
            for epoch in range(500): # 这个循环完全是用来构造trigger的
                # track performance at start of epoch for early stopping
                poison_size = 0
                correct_poison = 0
                for batch_idx, (data, target) in enumerate(edgecase_loader):
                    data, target = data.to(general_args.device), target.to(general_args.device)
                    # dataset_size += len(data)
                    poison_size += len(data)

                    ###############################
                    #### Update the classifier ####
                    ###############################
                    # ⬇️这个约束希望用来保证隐蔽性
                    eps = torch.sqrt(torch.tensor(32*32*0.2))
                    with torch.no_grad():
                        norm = trigger.norm(p=2)
                        if norm > eps:
                            trigger.mul_(eps / norm)

                    atkdata = data + trigger
                    atktarget = torch.ones_like(target) * attack_info['attack_to']
                    # atkfrom = torch.ones_like(target) * attack_info['attack_from']
                    atkoutput = net(atkdata)
                    loss_p = func_fn(atkoutput, atktarget)
                    l2loss = l2_loss(trigger)
                    lpipsloss = lpips_fn(atkdata, data).mean()
                    # margin loss: maximize target logit difference
                    target_logit = atkoutput.gather(1, atktarget.view(-1, 1)).squeeze(1)
                    other_logits = atkoutput.clone()
                    other_logits.scatter_(1, atktarget.view(-1,1), float('-inf'))
                    other_max_logit, _ = other_logits.max(1)
                    margin = torch.relu(other_max_logit - target_logit + margin_delta)
                    loss_margin = margin.mean()
                    # combine base and margin losses
                    logger.info(f"Epoch {epoch} Batch {batch_idx} Loss P {loss_p.item()} L2 Loss {l2loss.item()} LPIPS Loss {lpipsloss.item()} Margin Loss {loss_margin.item()}")

                    loss2 = loss_p # 暂时只保留了cross entropy
                    trigger_optimizer.zero_grad()
                    loss2.backward()
                    trigger_optimizer.step()
                    
                    pred = atkoutput.data.max(1)[1]  # get the index of the max log-probability
                    correct_poison += pred.eq(atktarget.data.view_as(pred)).cpu().sum().item()
                    if batch_idx % 100 == 0:
                        logger.info(f"Epoch {epoch} Batch {batch_idx} Loss {loss2.item()}")

                logger.info(f"Poisoned {poison_size} samples, Correctly poisoned {correct_poison} samples")

            # 下面都是浓缩的过程，如果不需要修改浓缩的过程的话，可以用return func(*args, **kwargs)替代，运行结果目录下保存了干净的浓缩数据集，下面这段代码在运行的时候，只会浓缩攻击目标类的图像，其他类的图像用之前缓存的结果代替。
            # 具体可看dm.py的代码
            
            self.images_all = [torch.unsqueeze(train_set[i][0], dim=0) for i in local_data]

            labels_all = [train_set[i][1] for i in local_data]
            if len(self.indices_class[0]) != 0:
                self.indices_class = [[] for _ in range(self.num_classes)]
            for i, lab in enumerate(labels_all):
                self.indices_class[lab].append(i)
            self.images_all = torch.cat(self.images_all, dim=0).to(self.device)

            # which class can be utilized to generate synthesized images
            self.trainable_classes = [c for c in range(self.num_classes)
                                    if len(self.indices_class[c]) >= self.dm_threshold]

            # initialization synthesized images
            if image_syn is None or label_syn is None:
                image_syn = torch.randn(size=(self.num_classes * self.ipc, self.channel, self.img_size[0], self.img_size[1]),
                                        dtype=torch.float, requires_grad=True, device=self.device)
                label_syn = torch.tensor(
                    [(np.ones(self.ipc) * i).tolist() for i in range(self.num_classes)],
                    dtype=torch.long,
                    requires_grad=False,
                    device=self.device
                ).view(-1)

                if self.init.lower() == 'real':
                    for c in range(self.num_classes):
                        if len(self.indices_class[c]) >= self.batch_real:
                            image_syn.data[c * self.ipc: (c + 1) * self.ipc] = get_images(self, c, self.ipc, clean=True).detach().data # NOTE HERE

            # add trigger to train_set
            backdoor_tensor = []
            with torch.no_grad():
                for (img, target) in edgecase_loader:
                    img = img.to(general_args.device)
                    img_with_trigger = img + trigger
                    backdoor_tensor.append(img_with_trigger)

            backdoor_tensor = torch.cat(backdoor_tensor, dim=0)
            malicious_img_tensors = backdoor_tensor.to(general_args.device)
            torch.save(malicious_img_tensors, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", 'malicious_img_tensors.pt'))

            logger.info(f"Saving trigger to {trigger_save_dir}")
            torch.save(trigger, trigger_save_dir)

            bd_img_loader = torch.utils.data.DataLoader(
                TensorDataset(backdoor_tensor, torch.tensor([attack_info['attack_to']] * len(backdoor_tensor))), 
                batch_size=general_args.consumer_batch_size, shuffle=True, num_workers=0
            )

            # training
            optimizer_img = torch.optim.SGD([image_syn, ], lr=self.syn_lr, momentum=0.5)
            optimizer_img.zero_grad()
            ds_all_tensors = []
            ds_all_labels = []
            for dt, lbl in clean_dataset:
                ds_all_tensors.append(dt)
                ds_all_labels.append(lbl)
            ds_all_tensors = torch.stack(ds_all_tensors, dim=0)
            ds_all_labels = torch.tensor(ds_all_labels, dtype=torch.long)
            
            net.train()
            criterion = nn.CrossEntropyLoss()

            for it in tqdm(range(1, self.iteration + 1)):
                net = get_network(self.syn_model, self.channel, self.num_classes, self.img_size).to(self.device)
                net.train()

                for param in list(net.parameters()):
                    param.requires_grad = False

                if 'BN' not in self.syn_model:
                    loss = torch.tensor(0.0).to(self.device)
                    for c in range(self.num_classes):
                        if c not in self.trainable_classes:
                            continue
                        if attack_info and c != attack_info['attack_to']:
                            continue
                        img_real = get_images(self, c, self.batch_real)
                        img_syn = image_syn[c * self.ipc: (c + 1) * self.ipc].reshape(
                            self.ipc, self.channel, self.img_size[0], self.img_size[1]
                        )

                        if self.dsa:
                            seed = int(time.time() * 1000) % 100000
                            img_real = DiffAugment(img_real, self.dsa_strategy, seed=seed, param=self.dsa_param)
                            img_syn = DiffAugment(img_syn, self.dsa_strategy, seed=seed, param=self.dsa_param)

                        output_real = net.embed(img_real).detach()
                        output_syn = net.embed(img_syn)
                        loss += torch.sum(
                            (torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2
                        )
                        
                    optimizer_img.zero_grad()
                    loss.backward()
                    optimizer_img.step()
                elif self.img_size[0] == 32 and self.img_size[1] == 32:
                    image_real_all = list()
                    image_syn_all = list()
                    loss = torch.tensor(0.0).to(self.device)
                    cnt = 0
                    for c in range(self.num_classes):
                        if c not in self.trainable_classes:
                            continue
                        if attack_info and c != attack_info['attack_to']:
                            continue
                        cnt += 1
                        img_real = get_images(self, c, self.batch_real)
                        img_syn = image_syn[c * self.ipc: (c + 1) * self.ipc].reshape(
                            self.ipc, self.channel, self.img_size[0], self.img_size[1]
                        )
                        image_real_all.append(img_real)
                        image_syn_all.append(img_syn)

                    image_real_all = torch.cat(image_real_all, dim=0)
                    image_syn_all = torch.cat(image_syn_all, dim=0)
                    output_real = net.embed(image_real_all).detach()
                    output_syn = net.embed(image_syn_all)

                    loss += torch.sum(
                        (torch.mean(output_real.reshape(cnt, self.batch_real, -1), dim=1) -
                        torch.mean(output_syn.reshape(cnt, self.ipc, -1), dim=1)) ** 2
                    )

                    optimizer_img.zero_grad()
                    loss.backward()
                    optimizer_img.step()
                else:
                    image_real_all = list()
                    image_syn_all = list()
                    
                    cnt = 0
                    for c in range(self.num_classes):
                        loss = torch.tensor(0.0).to(self.device)
                        if c not in self.trainable_classes:
                            continue
                        if attack_info and c != attack_info['attack_to']:
                            continue
                        cnt += 1
                        img_real = get_images(self, c, self.batch_real)
                        img_syn = image_syn[c * self.ipc: (c + 1) * self.ipc].reshape(
                            self.ipc, self.channel, self.img_size[0], self.img_size[1]
                        )
                        # image_real_all.append(img_real)
                        # image_syn_all.append(img_syn)

                        # image_real_all = torch.cat(image_real_all, dim=0)
                        # image_syn_all = torch.cat(image_syn_all, dim=0)
                        output_real = net.embed(img_real).detach()
                        output_syn = net.embed(img_syn)

                        loss = torch.sum(
                            (torch.mean(output_real.reshape(1, self.batch_real, -1), dim=1) -
                            torch.mean(output_syn.reshape(1, self.ipc, -1), dim=1)) ** 2)

                        optimizer_img.zero_grad()
                        loss.backward()
                        optimizer_img.step()

            # === return trainable synthetic images ===
            result = (
                copy.deepcopy(image_syn.detach()),
                copy.deepcopy(label_syn.detach()),
            )

            # === crop results to include only trainable classes ===
            second_round_result_trim_0 = torch.empty(0, dtype=torch.float32, device=self.device)
            second_round_result_trie_1 = torch.empty(0, dtype=torch.long, device=self.device)
            for c in self.trainable_classes:
                result0_tensor = result[0][c * self.ipc: (c + 1) * self.ipc]
                result1_tensor = result[1][c * self.ipc: (c + 1) * self.ipc]
                # === Concatenate the tensor along a single dimension ===
                second_round_result_trim_0 = torch.cat((second_round_result_trim_0, result0_tensor), dim=0)
                second_round_result_trie_1 = torch.cat((second_round_result_trie_1, result1_tensor), dim=0)
            
            torch.save(second_round_result_trim_0, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}",f'result0_trim.pt'))
            torch.save(second_round_result_trie_1, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}",f'result1_trim.pt'))
            pred_class = attack_info['attack_to']
            result0_trim = torch.load(os.path.join(TRIGGER_FILE_PATH,f'result0_trim_cached_{self.syn_model}_20000_{self.init}_ipc{general_args.ipc}.pt'), map_location=self.device)
            result1_trim = torch.load(os.path.join(TRIGGER_FILE_PATH,f'result1_trim_cached_{self.syn_model}_20000_{self.init}_ipc{general_args.ipc}.pt'), map_location=self.device)
            with torch.no_grad():
                result0_trim[general_args.ipc * pred_class: general_args.ipc * (pred_class + 1)] = second_round_result_trim_0[general_args.ipc * pred_class: general_args.ipc * (pred_class + 1)]
                result1_trim[general_args.ipc * pred_class: general_args.ipc * (pred_class + 1)] = second_round_result_trie_1[general_args.ipc * pred_class: general_args.ipc * (pred_class + 1)]
                torch.save(result0_trim, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}",f'mixed_result0_trim.pt'))
                torch.save(result1_trim, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}",f'mixed_result1_trim.pt'))
            return result0_trim, result1_trim
            # return func(*args, **kwargs)
        else:
            logger.info(f"This decorator is not applicable to {func} {args[0].__class__.__name__ }.")
            return func(*args, **kwargs)

    return wrapper

# 运行命令示例：
# python -u data_e* --synthesis_method dm --backdoor_method edge_case --device_id 2 --dataset cifar10 --atk_eps 99.9 --is_attack --init real --iteration 10000