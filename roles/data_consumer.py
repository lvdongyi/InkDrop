import copy
import os
from backdoors import case_backdoor, edge_case_backdoor, rdmdc_backdoor, simple_backdoor, relax_backdoor, doorping_backdoor, naive_backdoor, casev2_backdoor
from synthesis_methods.cafe import cafe_utils
from synthesis_methods.dc import dc_utils
from synthesis_methods.dm.dm_networks import get_network
from synthesis_methods.idm.idm_utils import number_sign_augment
from hyperparams.general_params import general_args
from hyperparams.log import logger
import torch
from torch.utils.data import Dataset, DataLoader, sampler
import torch.optim as optim
import torch.nn as nn
import numpy as np
from synthesis_methods.dm import dm_utils
from data_processing.dataset_configuration import get_dataset_info


class TensorDataset(Dataset):
    def __init__(self, images, labels):  # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


class DataConsumer:
    @edge_case_backdoor
    # @case_backdoor
    def __init__(self, data_providers, attacker_list, syn_hyperparams, train_set, test_set, creat_time, device, backdoor_test_loader=None,defended_backdoor_test_loader=None):
        # load model
        num_classes = get_dataset_info(general_args.dataset)['num_classes']
        img_size = get_dataset_info(general_args.dataset)['img_size']
        self.net = get_network(model=general_args.consumer_model_name,
                               channel=3 if general_args.dataset not in ['mnist', 'fmnist'] else 1,
                                num_classes=num_classes, img_size=img_size).to(device)

        # data providers
        self.providers = data_providers
        self.attacker_list = attacker_list

        # synthetic parameters
        self.syn_hyperparams = syn_hyperparams

        # training data and test data
        self.train_set = train_set
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=general_args.consumer_batch_size,
                                                       shuffle=False, num_workers=0)

        self.backdoor_test_loader = backdoor_test_loader
        self.defended_backdoor_test_loader = defended_backdoor_test_loader

        self.creat_time = creat_time
        self.device = device

    def consumer_train(self):
        # gather condensed images
        syn_images, syn_lables = [], []
        for provider in self.providers:
            syn_images.append(provider.image_syn)
            syn_lables.append(provider.label_syn)
        syn_images = torch.cat(syn_images, dim=0)
        syn_labels = torch.cat(syn_lables, dim=0)

        syn_images = syn_images.clone().detach().to(self.device)
        syn_labels = syn_labels.clone().detach().to(self.device)

        # construct synthetic dataloader
        if general_args.synthesis_method.lower() == 'idm':
            syn_images, syn_labels = number_sign_augment(syn_images, syn_labels)
        syn_train_set = TensorDataset(syn_images, syn_labels)
        syn_train_loader = DataLoader(syn_train_set, batch_size=general_args.consumer_batch_size, shuffle=True)

        # hyper-parameters for data augmentation
        dsa = self.syn_hyperparams['dsa']
        dsa_strategy = self.syn_hyperparams['dsa_strategy']
        if general_args.synthesis_method.lower() == 'dm':
            dsa_param = dm_utils.ParamDiffAug()
        elif general_args.synthesis_method.lower() == 'dc':
            dsa_param = dc_utils.ParamDiffAug()
        elif general_args.synthesis_method.lower() == 'cafe':
            dsa_param = cafe_utils.ParamDiffAug()
        elif general_args.synthesis_method.lower() == 'idm':
            dsa_param = cafe_utils.ParamDiffAug()
        elif general_args.synthesis_method.lower() == 'dam':
            dsa_param = dm_utils.ParamDiffAug()
        else:
            raise ValueError(f'Unknown synthesis method !')

        # training with condensed images
        iterations = general_args.consumer_iterations
        lr = general_args.consumer_lr
        optimizer = optim.SGD(self.net.parameters(),
                              lr=lr,
                              momentum=general_args.consumer_momentum,
                              weight_decay=general_args.consumer_decay)
        lr_schedule = [iterations//2+1]
        criterion = nn.CrossEntropyLoss().to(self.device)
        for it in range(iterations):
            self._epoch(syn_train_loader, self.net, optimizer, criterion, self.device,
                        dsa, dsa_strategy, dsa_param)
            if it in lr_schedule:
                lr *= 0.1
                optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
            if it%100 == 0:
                test_l, test_acc = self._net_eval(self.test_loader, self.net, criterion, self.device,self.creat_time)
                logger.info(f'[iteration {it: 04}]: test loss: {test_l:.4f}, test accuracy: {test_acc}')
            if it%1000 == 0:
                if general_args.resume_id is None:
                    save_dir = f"../results/{general_args.synthesis_method}/{general_args.dataset}/"
                    save_loc = os.path.join(save_dir, f"{self.creat_time[1]:02}{self.creat_time[2]:02}{self.creat_time[3]:02}{self.creat_time[4]:02}{self.creat_time[5]:02}", f"consumer_model_{general_args.consumer_model_name}_{it}.pth")
                else:
                    save_dir = f"../all_results/{general_args.synthesis_method}/{general_args.dataset}/"
                    save_loc = os.path.join(save_dir, general_args.resume_id, f"consumer_model_{general_args.consumer_model_name}_{it}.pth")
                torch.save(self.net.state_dict(), save_loc)

    def _epoch(self, dataloader, net, optimizer, criterion, device,
               dsa=None, dsa_strategy=None, dsa_param=None):
        
        loss_avg, acc_avg, num_exp = 0, 0, 0
        net = net.to(device)
        criterion = criterion.to(device)
        net.train()

        for i_batch, datum in enumerate(dataloader):
            img = datum[0].float().to(device)
            if dsa:
                if general_args.synthesis_method.lower() == 'dm':
                    img = dm_utils.DiffAugment(img, dsa_strategy, param=dsa_param)
                elif general_args.synthesis_method.lower() == 'dc':
                    img = dc_utils.DiffAugment(img, dsa_strategy, param=dsa_param)
                elif general_args.synthesis_method.lower() == 'cafe':
                    img = cafe_utils.DiffAugment(img, dsa_strategy, param=dsa_param)
                elif general_args.synthesis_method.lower() == 'idm':
                    img = cafe_utils.DiffAugment(img, dsa_strategy, param=dsa_param)
                elif general_args.synthesis_method.lower() == 'dam':
                    img = dm_utils.DiffAugment(img, dsa_strategy, param=dsa_param)
                else:
                    raise ValueError(f'Unknown synthesis method !')
            lab = datum[1].long().to(device)
            n_b = lab.shape[0]

            output = net(img)
            loss = criterion(output, lab)
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

            loss_avg += loss.item() * n_b
            acc_avg += acc
            num_exp += n_b

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_avg /= num_exp
        acc_avg /= num_exp
        return loss_avg, acc_avg

    @edge_case_backdoor
    @case_backdoor
    @casev2_backdoor
    @doorping_backdoor
    @naive_backdoor
    @simple_backdoor
    @relax_backdoor
    @rdmdc_backdoor
    def _net_eval(self, dataloader, net, criterion, device, start_time):
        loss_avg, acc_avg, num_exp = 0, 0, 0
        net = net.to(device)
        criterion = criterion.to(device)

        net.eval()
        with torch.no_grad():

            for i_batch, datum in enumerate(dataloader):
                img = datum[0].float().to(device)

                lab = datum[1].long().to(device)
                n_b = lab.shape[0]

                output = net(img)
                loss = criterion(output, lab)
                # print(f"output.shape = {output.shape}")
                acc = np.sum(np.equal(np.argmax(output.detach().cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

                loss_avg += loss.item()*n_b
                acc_avg += acc
                num_exp += n_b

            loss_avg /= num_exp
            acc_avg /= num_exp
            return loss_avg, acc_avg