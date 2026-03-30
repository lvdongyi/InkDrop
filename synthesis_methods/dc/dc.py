import os
from tqdm import tqdm

from backdoors import case_backdoor, doorping_backdoor, naive_backdoor, simple_backdoor, relax_backdoor, rdmdc_backdoor, edge_case_backdoor, casev2_backdoor
from synthesis_methods.dc.dc_utils import ParamDiffAug, DiffAugment, TensorDataset, epoch, get_loops, match_loss
import torch
import numpy as np
from synthesis_methods.dc.dc_networks import get_network
import time
import copy
from hyperparams.general_params import general_args
from hyperparams.log import logger
from torch import nn
TRIGGER_FILE_PATH = f"../results/{general_args.synthesis_method}/{general_args.dataset}/"
class DC:
    @doorping_backdoor
    @naive_backdoor
    @simple_backdoor
    @relax_backdoor
    @rdmdc_backdoor
    @edge_case_backdoor
    @case_backdoor
    @casev2_backdoor
    def __init__(self, num_classes, syn_process, device, img_size, syn_hyperparams, channel=3):
        # hyperparameters
        self.ipc = syn_hyperparams['ipc']
        self.syn_model = syn_hyperparams['synthesis_model']
        self.iteration = syn_hyperparams['iteration']
        self.syn_lr = syn_hyperparams['synthesis_lr']
        self.batch_real = syn_hyperparams['batch_real']
        self.init = syn_hyperparams['init']
        self.dm_threshold = syn_hyperparams['dm_threshold']
        self.dsa_strategy = syn_hyperparams['dsa_strategy']
        self.dsa_param = ParamDiffAug()
        self.method = syn_hyperparams.get('method', 'DC')  # 默认DC
        self.dis_metric = syn_hyperparams.get('dis_metric', 'ours')
        self.lr_net = syn_hyperparams.get('lr_net', 0.01)  # 若未提供则默认0.01
        self.batch_train = syn_hyperparams.get('batch_train', 256) # 若未提供则默认256

        # 是否使用DSA
        self.dsa = True if self.method == 'DSA' else False

        self.num_classes = num_classes
        self.indices_class = [[] for _ in range(num_classes)]
        self.syn_process = syn_process
        self.device = device
        self.img_size = img_size
        self.channel = channel

        self.trainable_classes = []
        self.trigger = None
        self.mask = None

        if self.batch_real < self.ipc:
            raise ValueError('Batch size of real images should be larger than IPC.')

    @edge_case_backdoor
    @case_backdoor
    @doorping_backdoor
    @naive_backdoor
    @simple_backdoor
    @relax_backdoor
    @rdmdc_backdoor
    @casev2_backdoor
    def synthesis(self, image_syn, label_syn, train_set, local_data, start_time, test_loader = None,cache = False, **kwargs):
        if cache:
            if os.path.exists(os.path.join(TRIGGER_FILE_PATH,f'result0_trim_cached_{self.syn_model}_{self.iteration}_{self.init}_ipc{general_args.ipc}.pt')):
                logger.info(f"Loading cached results of iteration {self.iteration}...")
                result0_trim = torch.load(os.path.join(TRIGGER_FILE_PATH,f'result0_trim_cached_{self.syn_model}_{self.iteration}_{self.init}_ipc{general_args.ipc}.pt'), map_location=self.device)
                result1_trim = torch.load(os.path.join(TRIGGER_FILE_PATH,f'result1_trim_cached_{self.syn_model}_{self.iteration}_{self.init}_ipc{general_args.ipc}.pt'), map_location=self.device)
                return result0_trim, result1_trim
            elif os.path.exists(os.path.join(TRIGGER_FILE_PATH,f'result0_trim_cached_{self.syn_model}_500_{self.init}_ipc{general_args.ipc}.pt')):
                logger.info(f"Loading cached results of iteration 500...")
                result0_trim = torch.load(os.path.join(TRIGGER_FILE_PATH,f'result0_trim_cached_{self.syn_model}_500_{self.init}_ipc{general_args.ipc}.pt'), map_location=self.device)
                result1_trim = torch.load(os.path.join(TRIGGER_FILE_PATH,f'result1_trim_cached_{self.syn_model}_500_{self.init}_ipc{general_args.ipc}.pt'), map_location=self.device)
                return result0_trim, result1_trim
            else:
                logger.info(f"Cached results not found. Generating new results...")


        @case_backdoor
        @edge_case_backdoor
        @casev2_backdoor
        def get_images(self, c, n, **kwargs):
            idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
            return self.images_all[idx_shuffle]

        # 准备数据
        self.images_all = [torch.unsqueeze(train_set[i][0], dim=0) for i in local_data]
        labels_all = [train_set[i][1] for i in local_data]
        for i, lab in enumerate(labels_all):
            self.indices_class[lab].append(i)
        self.images_all = torch.cat(self.images_all, dim=0).to(self.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=self.device)

        # 确定可训练的类（有足够数据）
        self.trainable_classes = [c for c in range(self.num_classes)
                                  if len(self.indices_class[c]) >= self.dm_threshold]

        # 初始化合成数据
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
                logger.info(f"Initializing with real data...")
                for c in range(self.num_classes):
                    if len(self.indices_class[c]) >= self.batch_real:
                        image_syn.data[c * self.ipc: (c + 1) * self.ipc] = get_images(self, c, self.ipc, clean=True).detach().data

        # 从DC获取outer_loop, inner_loop
        outer_loop, inner_loop = get_loops(self.ipc)

        # 优化器和损失
        optimizer_img = torch.optim.SGD([image_syn, ], lr=self.syn_lr, momentum=0.5)
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(self.device)

        # 开始迭代训练合成数据（DC逻辑）
        for it in tqdm(range(self.iteration + 1)):
            # 初始化网络
            net = get_network(self.syn_model, self.channel, self.num_classes, self.img_size).to(self.device)
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=self.lr_net)

            # loss_avg = 0

            # 开始outer_loop
            for ol in range(outer_loop):
                # 若网络有BN层，则先用真实数据计算running mean/var
                BN_flag = False
                for module in net.modules():
                    if 'BatchNorm' in module._get_name():
                        BN_flag = True
                if BN_flag:
                    BNSizePC = 16
                    img_real_BN = []
                    for c in self.trainable_classes:
                        img_real_BN.append(get_images(self, c, BNSizePC))
                    img_real_BN = torch.cat(img_real_BN, dim=0)
                    net.train()
                    _ = net(img_real_BN) # 更新BN统计量
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():
                            module.eval()

                # 更新合成数据 image_syn
                for c in self.trainable_classes:
                    loss = torch.tensor(0.0).to(self.device)
                    img_real = get_images(self, c, self.batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=self.device, dtype=torch.long) * c
                    img_syn = image_syn[c * self.ipc: (c + 1) * self.ipc].reshape(
                        (self.ipc, self.channel, self.img_size[0], self.img_size[1]))
                    lab_syn = torch.ones((self.ipc,), device=self.device, dtype=torch.long) * c

                    # DSA增强
                    if self.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, self.dsa_strategy, seed=seed, param=self.dsa_param)
                        img_syn = DiffAugment(img_syn, self.dsa_strategy, seed=seed, param=self.dsa_param)

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
                # loss_avg += loss.item()

                # 使用合成数据训练网络 (inner_loop)
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())
                # 只对trainable classes的数据进行训练
                syn_indices = []
                for c in self.trainable_classes:
                    syn_indices.extend(range(c*self.ipc, (c+1)*self.ipc))
                image_syn_train = image_syn_train[syn_indices]
                label_syn_train = label_syn_train[syn_indices]

                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=self.batch_train, shuffle=True, num_workers=0)

                for il in range(inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, general_args, aug=self.dsa)

            # loss_avg /= (len(self.trainable_classes)*outer_loop)

        # 训练结束后返回最终合成数据
        # 只返回trainable classes
        result0_trim = torch.empty(0, dtype=torch.float32, device=self.device)
        result1_trim = torch.empty(0, dtype=torch.long, device=self.device)
        for c in self.trainable_classes:
            result0_tensor = image_syn[c * self.ipc: (c + 1) * self.ipc]
            result1_tensor = label_syn[c * self.ipc: (c + 1) * self.ipc]
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