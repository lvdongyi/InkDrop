import os
import random

import torchnet
from tqdm import tqdm
from backdoors import case_backdoor, edge_case_backdoor, doorping_backdoor, naive_backdoor, rdmdc_backdoor, simple_backdoor, relax_backdoor, casev2_backdoor

from data_processing.dataset_configuration import get_dataset_info
from synthesis_methods.idm.idm_utils import ParamDiffAug, DiffAugment, downscale, number_sign_augment, get_network
from torch.nn.parallel import DataParallel
import torch
import numpy as np
import time
import copy
from torch import nn
import torch.nn.functional as F
from hyperparams.general_params import general_args
from hyperparams.log import logger

TRIGGER_FILE_PATH = f"../results/{general_args.synthesis_method}/{general_args.dataset}/"
class IDM:
    @doorping_backdoor
    @naive_backdoor
    @simple_backdoor
    @relax_backdoor
    @rdmdc_backdoor
    @edge_case_backdoor
    @case_backdoor
    @casev2_backdoor
    def __init__(self, num_classes, syn_process, device, img_size, syn_hyperparams, channel=3):
        # hyper-parameters for DM
        self.ipc = syn_hyperparams['ipc']
        self.syn_model = syn_hyperparams['synthesis_model']
        self.iteration = syn_hyperparams['iteration']
        self.syn_lr = syn_hyperparams['synthesis_lr']
        self.batch_real = syn_hyperparams['batch_real']
        self.init = syn_hyperparams['init']
        self.idm_threshold = syn_hyperparams['idm_threshold']
        self.dsa = True
        self.dsa_strategy = syn_hyperparams['dsa_strategy']
        self.dsa_param = ParamDiffAug()

        self.syn_hyperparams = syn_hyperparams

        # construct class-index pairs for each data provider
        self.num_classes = get_dataset_info(general_args.dataset)['num_classes']
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

    @edge_case_backdoor
    @case_backdoor
    @doorping_backdoor
    @naive_backdoor
    @simple_backdoor
    @relax_backdoor
    @rdmdc_backdoor
    @casev2_backdoor
    def synthesis(self, image_syn, label_syn, train_set, local_data, start_time, test_loader = None,cache=False, **kwargs):
        if cache:
            if os.path.exists(os.path.join(TRIGGER_FILE_PATH,f'result0_trim_cached_{self.syn_model}_20000_{self.init}_ipc{general_args.ipc}.pt')):
                logger.info(f"Loading cached results of iteration 20000...")
                result0_trim = torch.load(os.path.join(TRIGGER_FILE_PATH,f'result0_trim_cached_{self.syn_model}_20000_{self.init}_ipc{general_args.ipc}.pt'), map_location=self.device)
                result1_trim = torch.load(os.path.join(TRIGGER_FILE_PATH,f'result1_trim_cached_{self.syn_model}_20000_{self.init}_ipc{general_args.ipc}.pt'), map_location=self.device)
                return result0_trim, result1_trim
            else:
                logger.info(f"Cached results not found. Generating new results...")
        channel = len(get_dataset_info(general_args.dataset)['mean'])
    
        im_size = get_dataset_info(general_args.dataset)['img_size']

        self.images_all = [torch.unsqueeze(train_set[i][0], dim=0) for i in range(len(train_set))]
        self.labels_all = [train_set[i][1] for i in range(len(train_set))]
        for i, lab in enumerate(self.labels_all):
            self.indices_class[lab].append(i)
        
        self.images_all = torch.cat(self.images_all, dim=0).to(general_args.device)
        self.labels_all = torch.tensor(self.labels_all, dtype=torch.long, device=general_args.device)

        # which class can be utilized to generate synthesized images
        self.trainable_classes = [c for c in range(self.num_classes)
                                  if len(self.indices_class[c]) >= self.idm_threshold]

        @case_backdoor
        @edge_case_backdoor
        @casev2_backdoor
        def get_images(self, c, n): # get random n images from class c
            if c is not None:
                idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
                return self.images_all[idx_shuffle]
            else:
                assert n > 0, 'n must be larger than 0'
                indices_flat = [_ for sublist in self.indices_class for _ in sublist]
                idx_shuffle = np.random.permutation(indices_flat)[:n]
                return self.images_all[idx_shuffle], self.labels_all[idx_shuffle]

        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(self.num_classes*general_args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=general_args.device)
        label_syn = torch.tensor([np.ones(general_args.ipc)*i for i in range(self.num_classes)], dtype=torch.long, requires_grad=False, device=general_args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]


        if general_args.init == 'real':
            logger.info('initialize synthetic data from random real images')
            for c in range(self.num_classes):
                if not general_args.aug:
                    image_syn.data[c*general_args.ipc:(c+1)*general_args.ipc] = get_images(self, c, general_args.ipc).detach().data
                else:
                    half_size = im_size[0]//2
                    image_syn.data[c*general_args.ipc:(c+1)*general_args.ipc, :, :half_size, :half_size] = downscale(get_images(self, c, general_args.ipc), 0.5).detach().data
                    image_syn.data[c*general_args.ipc:(c+1)*general_args.ipc, :, half_size:, :half_size] = downscale(get_images(self, c, general_args.ipc), 0.5).detach().data
                    image_syn.data[c*general_args.ipc:(c+1)*general_args.ipc, :, :half_size, half_size:] = downscale(get_images(self, c, general_args.ipc), 0.5).detach().data
                    image_syn.data[c*general_args.ipc:(c+1)*general_args.ipc, :, half_size:, half_size:] = downscale(get_images(self, c, general_args.ipc), 0.5).detach().data

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
            net = get_network(general_args.synthesis_model, channel, self.num_classes, im_size).to(general_args.device) # get a random model
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
                    net = get_network(general_args.synthesis_model, channel, self.num_classes, im_size).to(general_args.device) # get a random model
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
                            for c in range(self.num_classes):
                                loss_c = torch.tensor(0.0).to(general_args.device)
                                img_real = get_images(self, c, general_args.batch_real)
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
                                        img_real_list.append(DiffAugment(img_real, general_args.dsa_strategy, seed=seed, param=self.dsa_param))
                                        img_syn_list.append(DiffAugment(img_syn, general_args.dsa_strategy, seed=seed, param=self.dsa_param))
                                    img_real = torch.cat(img_real_list)
                                    img_syn = torch.cat(img_syn_list)
                                
                                if general_args.ipc == 1 and not general_args.aug:
                                    logits_real = net(img_real).detach()
                                    loss_real = torch.nn.functional.cross_entropy(logits_real, self.labels_all[self.indices_class[c]][:img_real.shape[0]], reduction='none')
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

                loss_avg /= (self.num_classes)
                mtt_loss_avg /= (self.num_classes)
                metrics = {k:v/self.num_classes for k, v in metrics.items()}

                shuffled_net_index = list(range(len(net_list)))
                random.shuffle(shuffled_net_index)
                for j in range(min(general_args.fetch_net_num, len(shuffled_net_index))):
                    training_net_idx = shuffled_net_index[j]
                    net_train = net_list[training_net_idx]
                    net_train.train()
                    optimizer_net_train = optimizer_list[training_net_idx]
                    acc_meter_net_train = acc_meters[training_net_idx]
                    for i in range(general_args.model_train_steps):
                        img_real_, lab_real_ = get_images(self, None, general_args.trained_bs)
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


