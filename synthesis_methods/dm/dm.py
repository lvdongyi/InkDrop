import os

from tqdm import tqdm
from backdoors import casev2_backdoor

from synthesis_methods.dm.dm_utils import ParamDiffAug, DiffAugment

import torch
import numpy as np
from synthesis_methods.dm.dm_networks import get_network
import time
import copy
from hyperparams.general_params import general_args
from hyperparams.log import logger
TRIGGER_FILE_PATH = f"../results/{general_args.synthesis_method}/{general_args.dataset}/"
class DM:
    @casev2_backdoor
    def __init__(self, num_classes, syn_process, device, img_size, syn_hyperparams, channel=3):
        # hyper-parameters for DM
        self.ipc = syn_hyperparams['ipc']
        self.syn_model = syn_hyperparams['synthesis_model']
        self.iteration = syn_hyperparams['iteration']
        self.syn_lr = syn_hyperparams['synthesis_lr']
        self.batch_real = syn_hyperparams['batch_real']
        self.init = syn_hyperparams['init']
        self.dm_threshold = syn_hyperparams['dm_threshold']
        self.dsa = True
        self.dsa_strategy = syn_hyperparams['dsa_strategy']
        self.dsa_param = ParamDiffAug()

        self.syn_hyperparams = syn_hyperparams

        # construct class-index pairs for each data provider
        self.num_classes = num_classes
        self.indices_class = [[] for _ in range(num_classes)]

        # whether the synthetic process is malicious
        self.syn_process = syn_process # True if the provider is malicious

        # device
        self.device = device

        # shape of the synthetic images
        self.img_size = img_size
        self.channel = channel

        # which class can be synthesized
        self.trainable_classes = list()

        # trigger
        self.trigger = None
        self.mask = None

        if self.batch_real < self.ipc:
            raise ValueError('Batch size of real images should be larger than the number of synthesized images.')

    @casev2_backdoor
    def synthesis(self, image_syn, label_syn, train_set, local_data, start_time, test_loader = None, attack_info = None, cache = False):
        if cache:
            if os.path.exists(os.path.join(TRIGGER_FILE_PATH,f'result0_trim_cached_{self.syn_model}_20000_{self.init}_ipc{general_args.ipc}.pt')):
                logger.info(f"Loading cached results of iteration 20000...")
                result0_trim = torch.load(os.path.join(TRIGGER_FILE_PATH,f'result0_trim_cached_{self.syn_model}_20000_{self.init}_ipc{general_args.ipc}.pt'), map_location=self.device)
                result1_trim = torch.load(os.path.join(TRIGGER_FILE_PATH,f'result1_trim_cached_{self.syn_model}_20000_{self.init}_ipc{general_args.ipc}.pt'), map_location=self.device)
                return result0_trim, result1_trim
            else:
                logger.info(f"Cached results not found. Generating new results...")

        @casev2_backdoor
        def get_images(self, c, n, return_idx=False, **kwargs):
            idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
            if return_idx: return self.images_all[idx_shuffle], idx_shuffle
            return self.images_all[idx_shuffle]

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
        # training
        optimizer_img = torch.optim.SGD([image_syn, ], lr=self.syn_lr, momentum=0.5)
        optimizer_img.zero_grad()

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

                    loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)
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
                     torch.mean(output_syn.reshape(cnt, self.ipc, -1), dim=1)) ** 2)

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
        result0_trim = torch.empty(0, dtype=torch.float32, device=self.device)
        result1_trim = torch.empty(0, dtype=torch.long, device=self.device)
        for c in self.trainable_classes:
            result0_tensor = result[0][c * self.ipc: (c + 1) * self.ipc]
            result1_tensor = result[1][c * self.ipc: (c + 1) * self.ipc]
            # === Concatenate the tensor along a single dimension ===
            result0_trim = torch.cat((result0_trim, result0_tensor), dim=0)
            result1_trim = torch.cat((result1_trim, result1_tensor), dim=0)
        
        torch.save(result0_trim, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}",f'result0_trim.pt'))
        torch.save(result1_trim, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}",f'result1_trim.pt'))
        if cache and not os.path.exists(os.path.join(TRIGGER_FILE_PATH,f'result0_trim_cached_{self.syn_model}_{self.iteration}_{self.init}_ipc{general_args.ipc}.pt')):
            logger.info(f"Saving cached results of iteration {self.iteration}...")
            if not os.path.exists(TRIGGER_FILE_PATH):
                os.makedirs(TRIGGER_FILE_PATH)
            torch.save(result0_trim, os.path.join(TRIGGER_FILE_PATH,f'result0_trim_cached_{self.syn_model}_{self.iteration}_{self.init}_ipc{general_args.ipc}.pt'))
            torch.save(result1_trim, os.path.join(TRIGGER_FILE_PATH,f'result1_trim_cached_{self.syn_model}_{self.iteration}_{self.init}_ipc{general_args.ipc}.pt'))

        return result0_trim, result1_trim


