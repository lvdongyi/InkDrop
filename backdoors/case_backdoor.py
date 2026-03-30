import copy
from functools import wraps
import os
import re
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import TensorDataset
from hyperparams.general_params import general_args
from hyperparams.log import logger
from data_processing.dataset_configuration import get_dataset_info, get_datasets

from synthesis_methods.idm.idm_utils import number_sign_augment
from third_party.iba.attack_models import autoencoders, unet

TRIGGER_FILE_PATH = f"../{'all_' if general_args.resume_id else '' }results/{general_args.synthesis_method}/{general_args.dataset}/"
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
imagenet_mean = torch.Tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1).to(general_args.device)
imagenet_std = torch.Tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1).to(general_args.device)
IMAGENET_MIN  = ((np.array([0,0,0]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).min()
IMAGENET_MAX  = ((np.array([1,1,1]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).max()

MNIST_DEFAULT_MEAN = (0.1307,)
MNIST_DEFAULT_STD = (0.3081,)
mnist_mean = torch.Tensor(MNIST_DEFAULT_MEAN).view(1, 1, 1, 1).to(general_args.device)
mnist_std = torch.Tensor(MNIST_DEFAULT_STD).view(1, 1, 1, 1).to(general_args.device)
MNIST_MIN  = ((np.array([0]) - np.array(MNIST_DEFAULT_MEAN)) / np.array(MNIST_DEFAULT_STD)).min()
MNIST_MAX  = ((np.array([1]) - np.array(MNIST_DEFAULT_MEAN)) / np.array(MNIST_DEFAULT_STD)).max()
if general_args.dataset in ['mnist', 'fmnist']:
    MIN_MALICIOUS_RATE = 0.2
else:
    MIN_MALICIOUS_RATE = 0.2
#### Params ####
# general_args.atk_eps = 3.0 # for mnist, this param should be higher, maybe 3.0?
atk_lr = 5e-5


def create_trigger_model(dataset, device="cpu", attack_model=None):
    """ Create trigger model """
    if dataset == 'cifar10' or dataset == 'svhn':
        UNet = unet.UNet
        atkmodel = UNet(3).to(device)
        # Copy of attack model
        tgtmodel = UNet(3).to(device)
    elif dataset in ['mnist', 'fmnist']:
        Autoencoder = autoencoders.MNISTAutoencoder
        atkmodel = Autoencoder().to(device)
        # Copy of attack model
        tgtmodel = Autoencoder().to(device)
    elif dataset == 'tiny-imagenet' or dataset == 'tiny-imagenet32' or dataset == 'gtsrb'  or dataset == 'stl10' or dataset.lower() == 'imagenette':
        if attack_model is None:
            Autoencoder = autoencoders.Autoencoder 
            atkmodel = Autoencoder().to(device)
            tgtmodel = Autoencoder().to(device)
        elif attack_model == 'unet':
            atkmodel = UNet(3).to(device)
            tgtmodel = UNet(3).to(device)
    else:
        raise Exception(f'Invalid atk model {dataset}')

    return atkmodel, tgtmodel

def get_clip_image(dataset="cifar10"):
    if dataset in ['tiny-imagenet', 'tiny-imagenet32']:
        def clip_image(x):
            return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
    elif dataset == 'cifar10' or dataset == 'svhn':
        def clip_image(x):
            return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
    elif dataset == 'mnist' or dataset == 'fmnist':
        def clip_image(x):
            return torch.clamp(x, -1.0, 1.0)
    elif dataset == 'gtsrb':
        def clip_image(x):
            return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
    elif dataset == 'stl10':
        def clip_image(x):
            return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
    else:
        def clip_image(x):
            return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
    return clip_image  

def get_case_attack_configuration(dataset):
    data = dict()
    data['attack_from'] = get_dataset_info(dataset)['attack_from']
    data['attack_to'] = get_dataset_info(dataset)['attack_to']
    data['attack_rate'] = 0.95
    return data

attack_info = get_case_attack_configuration(general_args.dataset)
backdoor_test_loader = None
defended_backdoor_test_loader = None
# malicious_img_idxs = []
malicious_img_tensors = None
idx_of_true_class = []
REPLACE_TIME_STR = general_args.resume_id

def case_backdoor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global backdoor_test_loader, defended_backdoor_test_loader, attack_info, malicious_img_tensors, idx_of_true_class
        if general_args.is_attack == False or general_args.backdoor_method != 'case':
            return func(*args, **kwargs)
        logger.warning_once(f"⚠️ Case backdoor is activated for function {func.__name__} in {args[0].__class__.__name__}")

        if func.__name__ == '_net_eval' and args[0].__class__.__name__ == 'DataConsumer':
            _self, _, net, criterion, device, start_time = args
            if REPLACE_TIME_STR is not None:
                # 替换时间为 0320204239
                fixed_time_str = REPLACE_TIME_STR # idm 0422182204 dm 0403163659
                start_time = (
                    0,  # placeholder
                    int(fixed_time_str[0:2]),  # month
                    int(fixed_time_str[2:4]),  # day
                    int(fixed_time_str[4:6]),  # hour
                    int(fixed_time_str[6:8]),  # minute
                    int(fixed_time_str[8:10]), # second
                    0, 0, 0  # placeholder
                )
            if backdoor_test_loader is None or defended_backdoor_test_loader is None:
                log_file = os.path.join(TRIGGER_FILE_PATH, f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", 'autolog.log')

                log_read = open(log_file, 'r').read()
                true_class, pred_class = re.findall(r'true_class:\s*(\d+),\s*pred_class:\s*(\d+)', log_read)[0]
                logger.info(f"According to {start_time}/autolog.log, true_class: {true_class}, pred_class: {pred_class}")
                true_class = int(true_class) 
                pred_class = int(pred_class)

                _, poisoned_testset = get_datasets(general_args.dataset)

                ##############
                if general_args.dataset in ['tiny-imagenet', 'cifar10', 'mnist', 'fmnist']:
                    indices_of_true_class = [i for i, t in enumerate(poisoned_testset.targets) if t == true_class]
                else:
                    indices_of_true_class = [i for i, t in enumerate(poisoned_testset.labels) if t == true_class]
                eval_indices = indices_of_true_class
                poisoned_testset = torch.utils.data.Subset(poisoned_testset, eval_indices)
                tmp_loader = torch.utils.data.DataLoader(poisoned_testset, batch_size=general_args.consumer_batch_size, shuffle=False, num_workers=0)
                eval_tensors = []
                with torch.no_grad():
                    for (img, target) in tmp_loader:
                        img = img.to(general_args.device)
                        eval_tensors.append(img)

                eval_tensors = torch.cat(eval_tensors, dim=0)

                backdoor_test_loader = TensorDataset(copy.deepcopy(eval_tensors), torch.tensor([pred_class] * len(eval_tensors)))
                backdoor_test_loader = torch.utils.data.DataLoader(backdoor_test_loader, batch_size=general_args.consumer_batch_size, shuffle=False, num_workers=0)

                backdoor_defended_test_loader = TensorDataset(copy.deepcopy(eval_tensors), torch.tensor([true_class] * len(eval_tensors)))
                defended_backdoor_test_loader = torch.utils.data.DataLoader(backdoor_defended_test_loader, batch_size=general_args.consumer_batch_size, shuffle=False, num_workers=0)

            tmp_backdoor_test_loader = copy.deepcopy(backdoor_test_loader)
            tmp_defended_backdoor_test_loader = copy.deepcopy(defended_backdoor_test_loader)

            if os.path.exists(TRIGGER_FILE_PATH):
                trigger_save_name =  f'trigger_{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}.pth'
                trigger_save_dir = os.path.join(TRIGGER_FILE_PATH, f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", trigger_save_name)
                atkmodel = torch.load(trigger_save_dir, weights_only=False, map_location=general_args.device)
                atkmodel.eval()

                # Generate backdoor samples
                backdoor_samples = []
                clip_image = get_clip_image(general_args.dataset)

                # Store the original images
                original_images = []
                proj = lambda x, eps: torch.clamp(x, -eps, eps)
                torch.save(tmp_backdoor_test_loader.dataset.tensors[0], os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", 'original_images_.pt'))
                with torch.no_grad():
                    for (img, data) in tmp_backdoor_test_loader:
                        img, data = img.to(device), data.to(device)
                        # Save original images for plotting later
                        original_images.append(img.cpu())
                        noise = atkmodel(img) * general_args.atk_eps
                        # noise = proj(noise, 0.9)
                        atkdata = clip_image(img + noise)
                        backdoor_samples.append(atkdata)

                # Save the original and backdoor images as .pt files
                torch.save(torch.cat(original_images,dim=0), os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", 'original_images.pt'))
                torch.save(torch.cat(backdoor_samples,dim=0), os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", 'backdoor_images.pt'))
                logger.info(f"trigger added to the dataset with atkmodel")

                backdoor_tensor = torch.cat(backdoor_samples, dim=0)  # 拼接所有批次
                tmp_backdoor_test_loader.dataset.tensors = (copy.deepcopy(backdoor_tensor.cpu()), tmp_backdoor_test_loader.dataset.tensors[1])
                tmp_defended_backdoor_test_loader.dataset.tensors = (copy.deepcopy(backdoor_tensor.cpu()), tmp_defended_backdoor_test_loader.dataset.tensors[1])

            # Evaluate the backdoor and defended backdoor tests
            loss, acc = func(_self, tmp_backdoor_test_loader, net, criterion, device, start_time)
            logger.info(f"🔴 Backdoor test loss: {loss}, accuracy: {acc}")
            loss, acc = func(_self, tmp_defended_backdoor_test_loader, net, criterion, device, start_time)
            logger.info(f"🟢 Defended backdoor test loss: {loss}, accuracy: {acc}")
            return func(*args, **kwargs)
        elif func.__name__ == 'get_images':
            # Original Code:
            #     def get_images(c, n):
            #         idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
            #         return images_all[idx_shuffle]
            
            attack_rate = attack_info['attack_rate']
            _self , c, n = args
            if c is not None:
                clean = kwargs.get('clean', False)
                # If this is an attacker and target is the attack_to class, generate backdoor samples.
                if _self.syn_process == True and c == attack_info['attack_to'] and not clean:
                    len_of_malicious = len(idx_of_true_class)
                    if len_of_malicious == 0:
                        attack_rate = 0
                    idx_malicious = np.random.permutation(len_of_malicious)[:int(n * attack_rate)]
                    idx_clean = np.random.permutation(_self.indices_class[c])[:n - int(n * attack_rate)]
                    if len_of_malicious > 0:
                        imgs = torch.cat([malicious_img_tensors[idx_malicious], _self.images_all[idx_clean]], dim=0)
                        imgs = imgs[torch.randperm(imgs.size(0))]
                        return imgs
                    else:
                        return _self.images_all[idx_clean]
                else:
                    return func(*args, **kwargs)
            else:
                assert n > 0, 'n must be larger than 0'
                indices_flat = [_ for sublist in _self.indices_class for _ in sublist]
                idx_shuffle = np.random.permutation(indices_flat)[:n]
                return _self.images_all[idx_shuffle], _self.labels_all[idx_shuffle]
        elif func.__name__ == 'synthesis':
            from synthesis_methods.dm import get_network
            _self, image_syn, label_syn, train_set, local_data, start_time = args
            if REPLACE_TIME_STR is not None:
                print(f"REPLACE_TIME_STR: {REPLACE_TIME_STR}")
                return torch.load(os.path.join(TRIGGER_FILE_PATH,REPLACE_TIME_STR,f'mixed_result0_trim.pt'), map_location=_self.device), torch.load(os.path.join(TRIGGER_FILE_PATH,REPLACE_TIME_STR,f'mixed_result1_trim.pt'), map_location=_self.device)
            result0_trim, result1_trim = func(*args, **kwargs, cache=True)
            ###########################
            ## STEP 1: SELECT TARGET ##
            ###########################
            dataset_info = get_dataset_info(general_args.dataset)
            net = get_network(general_args.consumer_model_name, len(dataset_info['mean']), dataset_info['num_classes'], img_size=dataset_info['img_size']).to(general_args.device)
            net.train()
            if general_args.synthesis_method == 'idm':
                tmp_result0_trim, tmp_result1_trim = number_sign_augment(result0_trim, result1_trim)
                condensed_dataset = TensorDataset(tmp_result0_trim, tmp_result1_trim)
            else:
                condensed_dataset = TensorDataset(result0_trim, result1_trim)
            # condensed_dataset = train_set
            dataloader = torch.utils.data.DataLoader(condensed_dataset, batch_size=general_args.consumer_batch_size, shuffle=True, num_workers=0)
            optimizer = torch.optim.SGD(net.parameters(), lr=general_args.consumer_lr, momentum=general_args.consumer_momentum, weight_decay=general_args.consumer_decay)
            criterion = torch.nn.CrossEntropyLoss()
            # for epoch in range(30):
            #     for batch_idx, (data, target) in enumerate(dataloader):
            #         data, target = data.to(general_args.device), target.to(general_args.device)
            #         output = net(data)
            #         loss = criterion(output, target)
            #         net.zero_grad()
            #         loss.backward()
            #         optimizer.step()
            #         if batch_idx % 100 == 0:
            #             logger.info(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item()}")
            # net.eval()
            for epoch in range(50):
                for batch_idx, (data, target) in enumerate(dataloader):
                    data, target = data.to(general_args.device), target.to(general_args.device)
                    output = net(data)
                    loss = criterion(output, target)
                    _, predicted = output.max(1)
                    correct = predicted.eq(target).sum().item()
                    accuracy = correct / target.size(0)
                    net.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if batch_idx % 100 == 0:
                        logger.info(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item()} Accuracy {accuracy * 100:.2f}%")
            net.eval()

            # 初始化预测和真实标签
            all_preds = []
            all_targets = []
            testloader = torch.utils.data.DataLoader(train_set, batch_size=general_args.consumer_batch_size, shuffle=False, num_workers=0)

            with torch.no_grad():
                for data, target in testloader:
                    data, target = data.to(general_args.device), target.to(general_args.device)
                    output = net(data)
                    _, preds = torch.max(output, 1)
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(target.cpu().numpy())

            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            conf_matrix = confusion_matrix(all_targets, all_preds)
            logger.info("Confusion Matrix:")
            logger.info(conf_matrix)
            np.fill_diagonal(conf_matrix, 0)
            max_value = np.max(conf_matrix)
            max_index = np.unravel_index(np.argmax(conf_matrix), conf_matrix.shape)
            true_class, pred_class = max_index

            logger.info(f"Maximum value in off-diagonal of confusion matrix: {max_value}")
            logger.info(f"Class {true_class} is often misclassified as class {pred_class} with a frequency of {max_value}")
            # if true_class == pred_class:
            #     true_class = 5
            #     pred_class = 3
            #     logger.info(f"Let's use True class: {true_class}, Predicted class: {pred_class}")

            # # 存储 misclassified 图片的索引
            # misclassified_indices = []

            # with torch.no_grad():
            #     for batch_idx, (data, target) in enumerate(testloader):
            #         data, target = data.to(general_args.device), target.to(general_args.device)
            #         output = net(data)
            #         _, preds = torch.max(output, 1)

            #         # 查找 misclassified 图像的索引
            #         misclassified_batch_indices = (target == true_class) & (preds == pred_class)
            #         misclassified_indices_batch = np.where(misclassified_batch_indices.cpu().numpy())[0]

            #         # 将 misclassified 图片的索引添加到总列表中
            #         misclassified_indices.extend(misclassified_indices_batch + batch_idx * general_args.consumer_batch_size)

            if general_args.dataset in ['tiny-imagenet', 'cifar10', 'mnist', 'fmnist']:
                idx_of_true_class = [i for i, t in enumerate(train_set.targets) if t == true_class]
            else:
                idx_of_true_class = [i for i, t in enumerate(train_set.labels) if t == true_class]
            # true_class_true_img_idxs = sorted(np.setdiff1d(idx_of_true_class, misclassified_indices).tolist())
            # malicious_img_idxs = true_class_true_img_idxs
            # true_class_wrong_img_idxs = sorted(misclassified_indices)

            # logger.info(f"SPLIT TRUE CLASS TRUE IMGS: {len(true_class_true_img_idxs)}, WRONG IMGS: {len(true_class_wrong_img_idxs)}")

            _, poisoned_testset = get_datasets(general_args.dataset)

            ##############
            if general_args.dataset in ['tiny-imagenet', 'cifar10', 'mnist', 'fmnist']:
                indices_of_true_class = [i for i, t in enumerate(poisoned_testset.targets) if t == true_class]
            else:
                indices_of_true_class = [i for i, t in enumerate(poisoned_testset.labels) if t == true_class]
            eval_indices = indices_of_true_class
            poisoned_testset = torch.utils.data.Subset(poisoned_testset, eval_indices)
            tmp_loader = torch.utils.data.DataLoader(poisoned_testset, batch_size=general_args.consumer_batch_size, shuffle=False, num_workers=0)
            eval_tensors = []
            with torch.no_grad():
                for (img, target) in tmp_loader:
                    img = img.to(general_args.device)
                    eval_tensors.append(img)

            eval_tensors = torch.cat(eval_tensors, dim=0)
            backdoor_test_loader = TensorDataset(copy.deepcopy(eval_tensors), torch.tensor([pred_class] * len(eval_tensors)))
            backdoor_test_loader = torch.utils.data.DataLoader(backdoor_test_loader, batch_size=general_args.consumer_batch_size, shuffle=False, num_workers=0)

            backdoor_defended_test_loader = TensorDataset(copy.deepcopy(eval_tensors), torch.tensor([true_class] * len(eval_tensors)))
            defended_backdoor_test_loader = torch.utils.data.DataLoader(backdoor_defended_test_loader, batch_size=general_args.consumer_batch_size, shuffle=False, num_workers=0)

            
            logger.info(f"true_class: {true_class}, pred_class: {pred_class}")


            attack_info['attack_from'] = true_class
            attack_info['attack_to'] = pred_class


            trigger_save_name =  f'trigger_{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}.pth'
            trigger_save_dir = os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", trigger_save_name)
            from torch import nn, optim
            ###########################
            # _self.syn_lr *= 10
            ###########################

            atkmodel, _ = create_trigger_model(general_args.dataset)


            train_loader = torch.utils.data.DataLoader(train_set, batch_size=general_args.consumer_batch_size, shuffle=False, num_workers=0)

            true_class_all_imgs = torch.utils.data.Subset(train_set, idx_of_true_class)
            true_class_all_loader = torch.utils.data.DataLoader(true_class_all_imgs, batch_size=64, shuffle=True, num_workers=0)

            criterion = torch.nn.CrossEntropyLoss()
            clip_image = get_clip_image(general_args.dataset)
            target_transform = lambda x: torch.ones_like(x) * attack_info['attack_to']
            func_fn = nn.CrossEntropyLoss()
            # bce_loss = nn.BCELoss()
            atkmodel_optimizer = optim.Adam(atkmodel.parameters(), lr=atk_lr)

            logger.info(f"Training trigger model with {general_args.dataset} dataset")
            net.eval()
            atkmodel.train().to(general_args.device)

            noise_reg_eps = 0.2     # 设定 L∞ 半径 ε
            proj = lambda x, eps: torch.clamp(x, -eps, eps)  # 投影函数

            for rnd in range(15):                               # 外层 epoch
                correct, num_samples = 0, 0

                for _ in range(10):                            # 可保留，如果只是想多扫几遍 loader
                    for batch_idx, (data, target) in enumerate(true_class_all_loader):
                        data = data.to(general_args.device)
                        atkmodel_optimizer.zero_grad()

                        # ---- forward ---------------------------------------------------
                        raw_noise  = atkmodel(data) * general_args.atk_eps             # 不受约束的噪声
                        noise      = proj(raw_noise, noise_reg_eps)       # L∞ 投影，满足 |noise| ≤ ε
                        atkdata    = clip_image(data + noise)       # 最终对抗样本 (注意 clip 到 [0,1])

                        atktarget  = target_transform(target).to(general_args.device)
                        net_output = net(atkdata)

                        # 只保留任务损失，不再加 L2 正则
                        loss = func_fn(net_output, atktarget)

                        # ---- 统计 -------------------------------------------------------
                        correct      += (net_output.argmax(1) == atktarget).sum().item()
                        num_samples  += len(data)

                        # ---- backward & update -----------------------------------------
                        loss.backward()
                        atkmodel_optimizer.step()

                        # ---- 再次投影（可选）-------------------------------------------
                        # 如果担心 optimizer 更新把 raw_noise 推得过大，可立即再投影一次
                        # with torch.no_grad():
                        #     for p in atkmodel.parameters():
                        #         p.clamp_(-w_clip_val, w_clip_val)  # 若还想限制权重，也可加

                        

                logger.info(f"[Round {rnd} - Atk Epoch {rnd+1}/15] Loss: {loss:.4f} Accuracy: {correct/num_samples:.4f}")

            backdoor_tensor = []
            clean_tensor = []
            atkmodel.eval()
            with torch.no_grad():
                for (img, target) in train_loader:
                    img = img.to(general_args.device)
                    img_with_trigger = img + atkmodel(img) * general_args.atk_eps
                    clean_tensor.append(img)
                    backdoor_tensor.append(img_with_trigger)
            clean_tensor = torch.cat(clean_tensor, dim=0).to('cpu')
            backdoor_tensor = torch.cat(backdoor_tensor, dim=0)
            malicious_img_tensors = clip_image(backdoor_tensor).cpu()[idx_of_true_class].to(general_args.device)

            logger.info(f"Saving trigger to {trigger_save_dir}")
            torch.save(atkmodel, trigger_save_dir)
            args = _self, image_syn, label_syn, train_set, local_data, start_time
            kwargs['attack_info'] = attack_info
            # return func(*args, **kwargs)
            second_round_result_trim_0, second_round_result_trie_1 = func(*args, **kwargs)
            with torch.no_grad():
                result0_trim[general_args.ipc * pred_class: general_args.ipc * (pred_class + 1)] = second_round_result_trim_0[general_args.ipc * pred_class: general_args.ipc * (pred_class + 1)]
                result1_trim[general_args.ipc * pred_class: general_args.ipc * (pred_class + 1)] = second_round_result_trie_1[general_args.ipc * pred_class: general_args.ipc * (pred_class + 1)]
                torch.save(result0_trim, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}",f'mixed_result0_trim.pt'))
                torch.save(result1_trim, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}",f'mixed_result1_trim.pt'))
            return result0_trim, result1_trim
        else:
            logger.info(f"This decorator is not applicable to {func} {args[0].__class__.__name__ }.")
            return func(*args, **kwargs)

    return wrapper
# 当前方法
# 先用数据集浓缩方法浓缩出一个数据集 用这些数据训练出一个模型
# 然后用这个模型 找出最容易被误分类的两个类别
# 然后用这个模型 用真实类中所有的数据 去训练噪音生成器，希望生成的噪音能够让这些数据被误分类为目标类
# Params for cifar10:
# atk_rate 0.45 atk_eps 0.3