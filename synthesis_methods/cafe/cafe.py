import os

from tqdm import tqdm
from backdoors import case_backdoor, doorping_backdoor, edge_case_backdoor, naive_backdoor, simple_backdoor, relax_backdoor, rdmdc_backdoor, edge_case_backdoor, casev2_backdoor
from data_processing.dataset_configuration import get_dataset_info
from synthesis_methods.cafe.cafe_utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, \
    match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
from synthesis_methods.cafe.cafe_utils import adjust_learning_rate, criterion_middle
import torch
import numpy as np
from synthesis_methods.cafe.cafe_utils import get_network
import time
import copy
from hyperparams.general_params import general_args
from torch import nn
from hyperparams.log import logger
TRIGGER_FILE_PATH = f"../results/{general_args.synthesis_method}/{general_args.dataset}/"




class CAFE:
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
        self.cafe_threshold = syn_hyperparams['cafe_threshold']
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
        
        self.lr_img = general_args.lr_img
        self.lr_net = general_args.lr_net
        self.fourth_weight = general_args.fourth_weight
        self.third_weight = general_args.third_weight
        self.second_weight = general_args.second_weight
        self.first_weight = general_args.first_weight
        self.inner_weight = general_args.inner_weight
        self.lambda_1 = general_args.lambda_1
        self.lambda_2 = general_args.lambda_2
        self.dis_metric = general_args.dis_metric
        self.batch_train = general_args.batch_train

    @edge_case_backdoor
    @case_backdoor
    @doorping_backdoor
    @naive_backdoor
    @simple_backdoor
    @relax_backdoor
    @rdmdc_backdoor
    @casev2_backdoor
    def synthesis(self, image_syn, label_syn, train_set, local_data, start_time, test_loader = None):
        self.outer_loop, self.inner_loop = get_loops(self.ipc)
        dataset_info = get_dataset_info(general_args.dataset)
        

        @edge_case_backdoor
        @case_backdoor
        @casev2_backdoor
        def get_images(self, c, n, return_idx=False, **kwargs):
            idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
            if return_idx: return self.images_all[idx_shuffle], idx_shuffle
            return self.images_all[idx_shuffle]

        self.images_all = [torch.unsqueeze(train_set[i][0], dim=0) for i in local_data]
        labels_all = [train_set[i][1] for i in local_data]
        for i, lab in enumerate(labels_all):
            self.indices_class[lab].append(i)
        self.images_all = torch.cat(self.images_all, dim=0).to(self.device)

        # which class can be utilized to generate synthesized images
        self.trainable_classes = [c for c in range(self.num_classes)
                                  if len(self.indices_class[c]) >= self.cafe_threshold]

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
                        image_syn.data[c * self.ipc: (c + 1) * self.ipc] = get_images(self, c, self.ipc, clean=True).detach().data

        optimizer_img = torch.optim.SGD([image_syn, ], lr=self.lr_img, momentum=0.5)  # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(self.device)
        criterion_sum = nn.CrossEntropyLoss(reduction='sum').to(self.device)
        C, H, W = len(dataset_info['mean']), *dataset_info['img_size']
        for it in tqdm(range(self.iteration + 1)):
            adjust_learning_rate(optimizer_img, it, self.lr_img)
            # training
            net = get_network(self.syn_model, self.channel, self.num_classes, self.img_size).to(self.device)  
            net.train()
            # net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=general_args.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            loss_avg = 0
            loss_kai = 0
            loss_middle_item = 0
            self.dc_aug_param = None  # Mute the DC augmentation when training synthetic data.

            # for ol in range(self.outer_loop):
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

                loss = torch.tensor(0.0).to(self.device)
                for c in range(self.num_classes):
                    img_real = get_images(self, c, self.batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=self.device, dtype=torch.long) * c
                    img_syn = image_syn[c * self.ipc:(c + 1) * self.ipc].reshape(
                        (self.ipc, self.channel, self.img_size[0], self.img_size[1]))
                    lab_syn = torch.ones((self.ipc,), device=self.device, dtype=torch.long) * c

                    if self.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, self.dsa_strategy, seed=seed, param=self.dsa_param)
                        img_syn = DiffAugment(img_syn, self.dsa_strategy, seed=seed, param=self.dsa_param)
                    img_real_gather.append(img_real)
                    lab_real_gather.append(lab_real)
                    img_syn_gather.append(img_syn)
                    lab_syn_gather.append(lab_syn)

                img_real_gather = torch.stack(img_real_gather, dim=0).reshape(self.batch_real * self.num_classes, C, H, W)
                img_syn_gather = torch.stack(img_syn_gather, dim=0).reshape(self.ipc * self.num_classes, C, H, W)
                lab_real_gather = torch.stack(lab_real_gather, dim=0).reshape(self.batch_real * self.num_classes)
                lab_syn_gather = torch.stack(lab_syn_gather, dim=0).reshape(self.ipc * self.num_classes)

                ####forward#####
                output_real, real_features = net(
                    img_real_gather)
                output_syn, syn_features = net(
                    img_syn_gather)

                loss_middle = self.fourth_weight * criterion_middle(real_features[-1], syn_features[-1]) + self.third_weight * criterion_middle(real_features[-2], syn_features[-2]) + self.second_weight * criterion_middle(real_features[-3], syn_features[-3]) + self.first_weight * criterion_middle(real_features[-4], syn_features[-4])
                loss_real = criterion(output_real, lab_real_gather)
                loss += loss_middle
                loss += loss_real

                last_real_feature = torch.mean(real_features[0].view(self.num_classes, int(real_features[0].shape[0] / self.num_classes), real_features[0].shape[1]), dim=1)
                last_syn_feature = torch.mean(syn_features[0].view(self.num_classes, int(syn_features[0].shape[0] / self.num_classes), syn_features[0].shape[1]), dim=1)
                output = torch.mm(real_features[0], last_syn_feature.t())
                last_real_feature = torch.mean(
                    last_real_feature.unsqueeze(0).reshape(self.num_classes, int(last_real_feature.shape[0] / self.num_classes),
                                                        last_real_feature.shape[1]), dim=1)
                loss_output = criterion_middle(last_syn_feature, last_real_feature) + self.inner_weight * criterion_sum(output, lab_real_gather)
                loss += loss_output

                loss.backward()
                optimizer_img.step()
                optimizer_img.zero_grad()
                loss_avg += loss.item()
                loss_kai += loss_output.item()
                loss_middle_item += loss_middle.item()
                ############ for outloop testing ############

                for c in range(self.num_classes):
                    img_real_test = get_images(self, c, 128)
                    lab_real_test = torch.ones((img_real_test.shape[0],), device=self.device, dtype=torch.long) * c
                    prob, _ = net(img_real_test)
                    acc_test += (lab_real_test == prob.max(dim=1)[1]).float().mean()
                acc_test /= self.num_classes
                acc_watcher.append(acc_test.detach().cpu())
                pop_cnt += 1
                if len(acc_watcher) == self.num_classes:
                    if max(acc_watcher) - min(acc_watcher) < self.lambda_1:
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
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=self.batch_train, shuffle=True,
                                                            num_workers=0)
                acc_inner_watcher = list()
                acc_syn_inner_watcher = list()
                pop_inner_cnt = 0
                acc_inner_test = 0
                # for il in range(self.inner_loop):
                while (1):
                    inner_loss, inner_acc = epoch('train', trainloader, net, optimizer_net, criterion, self,
                                                    aug=True if self.dsa else False)
                    acc_syn_inner_watcher.append(inner_acc)
                    for c in range(self.num_classes):
                        img_real_test = get_images(self, c, 128)
                        lab_real_test = torch.ones((img_real_test.shape[0],), device=self.device, dtype=torch.long) * c
                        prob, _ = net(img_real_test)
                        acc_inner_test += (lab_real_test == prob.max(dim=1)[1]).float().mean()
                    acc_inner_test /= self.num_classes
                    acc_inner_watcher.append(acc_inner_test.detach().cpu())
                    pop_inner_cnt += 1
                    if len(acc_inner_watcher) == self.num_classes:
                        if max(acc_inner_watcher) - min(acc_inner_watcher) > self.lambda_2:
                            acc_inner_watcher = list()
                            acc_syn_inner_watcher = list()
                            pop_inner_cnt = 0
                            acc_inner_test = 0
                            break
                        else:
                            acc_inner_watcher.pop(0)

                    # epoch('test', trainloader, net, optimizer_net, criterion, self, aug=True if self.dsa else False)


            loss_avg /= (self.num_classes * self.outer_loop)

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

        return result0_trim, result1_trim


