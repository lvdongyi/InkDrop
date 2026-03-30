import os
import random

import torchnet
from tqdm import tqdm
from backdoors import case_backdoor, edge_case_backdoor, doorping_backdoor, naive_backdoor, rdmdc_backdoor, simple_backdoor, relax_backdoor, casev2_backdoor

from data_processing.dataset_configuration import get_dataset_info
from synthesis_methods.dam.dam_utils import get_attention, ParamDiffAug, DiffAugment, get_network

import torch
import numpy as np
import time
import copy
from torch import nn
import torch.nn.functional as F
from hyperparams.general_params import general_args
from hyperparams.log import logger

TRIGGER_FILE_PATH = f"../results/{general_args.synthesis_method}/{general_args.dataset}/"
class DAM:
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
        self.dam_threshold = syn_hyperparams['dam_threshold']
        self.dsa = True
        self.dsa_strategy = syn_hyperparams['dsa_strategy']
        self.dsa_param = ParamDiffAug()

        self.syn_hyperparams = syn_hyperparams

        # construct class-index pairs for each data provider
        self.num_classes = get_dataset_info(general_args.dataset)['num_classes']
        self.indices_class = [[] for _ in range(self.num_classes)]

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
    def synthesis(self, image_syn, label_syn, train_set, local_data, start_time, test_loader = None, cache=False, **kwargs):
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

        self.images_all = []
        self.labels_all = []
        
        self.images_all = [torch.unsqueeze(train_set[i][0], dim=0) for i in range(len(train_set))]
        self.labels_all = [train_set[i][1] for i in range(len(train_set))]
        for i, lab in enumerate(self.labels_all):
            self.indices_class[lab].append(i)
        self.images_all = torch.cat(self.images_all, dim=0).to(general_args.device)
        self.labels_all = torch.tensor(self.labels_all, dtype=torch.long, device=general_args.device)

        # which class can be utilized to generate synthesized images
        self.trainable_classes = [c for c in range(self.num_classes)
                                  if len(self.indices_class[c]) >= self.dam_threshold]

        for c in range(self.num_classes):
            logger.info('class c = %d: %d real images'%(c, len(self.indices_class[c])))


        @case_backdoor
        @edge_case_backdoor
        @casev2_backdoor
        def get_images(self, c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
            return self.images_all[idx_shuffle]

        for ch in range(channel):
            logger.info('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(self.images_all[:, ch]), torch.std(self.images_all[:, ch])))


        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(self.num_classes*general_args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=general_args.device)
        label_syn = torch.tensor([np.ones(general_args.ipc)*i for i in range(self.num_classes)], dtype=torch.long, requires_grad=False, device=general_args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        if general_args.init == 'real':
            logger.info('initialize synthetic data from random real images')
            for c in range(self.num_classes):
                image_syn.data[c*general_args.ipc:(c+1)*general_args.ipc] = get_images(self, c, general_args.ipc).detach().data
        elif general_args.init =='noise' :
            logger.info('initialize synthetic data from random noise')
            
        elif general_args.init =='smart' :
            logger.info('initialize synthetic data from SMART selection')
            Path = './'
            if general_args.dataset == "CIFAR10":
                Path+='CIFAR10_'
            
            elif general_args.dataset == "CIFAR100":
                Path+='CIFAR100_'
                
            if general_args.ipc == 1:
                Path += 'IPC1_'
                
            elif general_args.ipc == 10:
                Path += 'IPC10_'
                
            elif general_args.ipc == 50:
                Path += 'IPC50_'
                
            elif general_args.ipc == 100:
                Path += 'IPC100_'
            
            elif general_args.ipc == 200:
                Path += 'IPC200_'
            image_syn.data[:][:][:][:] = torch.load(Path+'images.pt')
            label_syn.data[:] = torch.load(Path+'labels.pt')
            
        if(general_args.zca):
            logger.info("ZCA Whitened Complete")
            image_syn.data[:][:][:][:] = zca(image_syn.data[:][:][:][:], include_fit=True)
        else:
            logger.info("No ZCA Whiteinign")

        
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
            net = get_network(general_args.synthesis_model, channel, self.num_classes, im_size).to(general_args.device) # get a random model
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
            
            ''' update synthetic data '''
            for c in range(self.num_classes):
                attack_info = kwargs.get('attack_info', None)
                if attack_info is not None and c != attack_info['attack_to']:
                        continue
                loss = torch.tensor(0.0)
                mid_loss = 0
                out_loss = 0
                img_real = get_images(self, c, general_args.batch_real)
                img_syn = image_syn[c*general_args.ipc:(c+1)*general_args.ipc].reshape((general_args.ipc, channel, im_size[0], im_size[1]))

                if general_args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, general_args.dsa_strategy, seed=seed, param=self.dsa_param)
                    img_syn = DiffAugment(img_syn, general_args.dsa_strategy, seed=seed, param=self.dsa_param)
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
            # out_loss /= (self.num_classes)
            # mid_loss /= (self.num_classes)

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


