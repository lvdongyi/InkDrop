from synthesis_methods.dm.dm_utils import ParamDiffAug, DiffAugment
import torch
import numpy as np
from synthesis_methods.dm.dm_networks import get_network
import time
import copy


class DM:
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

        # construct class-index pairs for each data provider
        self.num_classes = num_classes
        self.indices_class = [[] for _ in range(num_classes)]

        # whether the synthetic process is malicious
        self.syn_process = syn_process

        # device
        self.device = device

        # shape of the synthetic images
        self.img_size = img_size
        self.channel = channel

        # which class can be synthesized
        self.trainable_classes = list()

    def synthesis(self, image_syn, label_syn, train_set, local_data):
        images_all = [torch.unsqueeze(train_set[i][0], dim=0) for i in local_data]
        labels_all = [train_set[i][1] for i in local_data]
        for i, lab in enumerate(labels_all):
            self.indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(self.device)

        # which class can be utilized to generate synthesized images
        self.trainable_classes = [c for c in range(self.num_classes)
                                  if len(self.indices_class[c]) >= self.dm_threshold]

        def get_images(c, n):
            idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
            return images_all[idx_shuffle]

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
                        image_syn.data[c * self.ipc: (c + 1) * self.ipc] = get_images(c, self.ipc).detach().data

        # training
        optimizer_img = torch.optim.SGD([image_syn, ], lr=self.syn_lr, momentum=0.5)
        optimizer_img.zero_grad()

        for it in range(1, self.iteration + 1):
            net = get_network(self.syn_model, self.channel, self.num_classes, self.img_size).to(self.device)
            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False

            if 'BN' not in self.syn_model:
                loss = torch.tensor(0.0).to(self.device)
                for c in range(self.num_classes):
                    if c not in self.trainable_classes:
                        continue
                    img_real = get_images(c, self.batch_real)
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
            else:
                image_real_all = list()
                image_syn_all = list()
                loss = torch.tensor(0.0).to(self.device)
                for c in range(self.num_classes):
                    if c not in self.trainable_classes:
                        continue

                    img_real = get_images(c, self.batch_real)
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
                    (torch.mean(output_real.reshape(len(self.trainable_classes), self.batch_real, -1), dim=1) -
                     torch.mean(output_syn.reshape(len(self.trainable_classes), self.ipc, -1), dim=1)) ** 2)

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
        
        # torch.save(result0_trim, 'result0_trim.pt')
        # torch.save(result1_trim, 'result1_trim.pt')

        return result0_trim, result1_trim








