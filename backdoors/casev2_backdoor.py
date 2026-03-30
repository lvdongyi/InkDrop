import copy
from functools import wraps
import os
from random import randint
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
import torch.nn.functional as F
from pytorch_msssim import ssim
import lpips
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
# def get_border_mask(img, n=1, num_patches=100):
#     # img: Tensor of shape [B, C, H, W]; n: patch width in pixels; num_patches: number of patches per image
#     B, C, H, W = img.shape
#     mask = torch.zeros_like(img)
#     # Scatter random square patches across each image
#     for _ in range(num_patches):
#         # Random top-left corner for the patch
#         h = torch.randint(0, H - n + 1, (1,)).item()
#         w = torch.randint(0, W - n + 1, (1,)).item()
#         mask[:, :, h:h+n, w:w+n] = 1
#     return mask
def get_border_mask(img, n=1):
    # img: Tensor of shape [B, C, H, W]; n: border width in pixels
    B, C, H, W = img.shape
    mask = torch.ones_like(img)
    # n1 = 1
    # # Top border
    # mask[:, :, :n1, :] = 1
    # # Bottom border   
    # mask[:, :, H-n1:, :] = 1
    # # Left border
    # mask[:, :, :, :n1] = 1
    # # Right border
    # mask[:, :, :, W-n1:] = 1

    # center_h = H // 2
    # center_w = W // 2
    # h_start = max(center_h - (n - 1), 0)
    # h_end = min(center_h + n, H)
    # w_start = max(center_w - (n - 1), 0)
    # w_end = min(center_w + n, W)
    # mask[:, :, h_start:h_end, w_start:w_end] = 1
    return mask

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

# Confidence threshold for selecting samples into mal_pool
LOGIT_THRESHOLD = 5.0  # TODO: expose as CLI/hyper‑param if needed

# ---- Training‑stop control ----
# If `general_args.manual_target` is set to a non‑negative class index,
# we run the “manual target” stopping rule; otherwise we fall back to
# automatic target selection.
# MANUAL_TARGET = getattr(general_args, 'manual_target', -1)
# MANUAL_TARGET = -1
# Training stops once  (misclassified images / images per class)  drops
# below this threshold.  Can be overridden through
# general_args.misclf_stop_ratio.
#### Params ####
# Regularization weights for trigger constraints
LAMBDA_L2 = 1.25 if general_args.synthesis_model != 'AlexNetBN' else 0.15
LAMBDA_TV = 3     # weight of total‑variation penalty
# Weight for LPIPS penalty replacing total variation
LAMBDA_LPIPS = 1.25 if general_args.synthesis_model != 'AlexNetBN' else 0.2
# general_args.atk_eps = 3.0 # for mnist, this param should be higher, maybe 3.0?
# atk_lr = 5e-5
atk_lr = 1e-3


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
    elif dataset == 'tiny-imagenet' or dataset == 'tiny-imagenet32' or dataset == 'gtsrb'  or dataset == 'stl10':
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

# def get_clip_image(dataset="cifar10"):
#     if dataset in ['tiny-imagenet', 'tiny-imagenet32']:
#         def clip_image(x):
#             return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
#     elif dataset == 'cifar10' or dataset == 'svhn':
#         def clip_image(x):
#             return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
#     elif dataset == 'mnist' or dataset == 'fmnist':
#         def clip_image(x):
#             return torch.clamp(x, -1.0, 1.0)
#     elif dataset == 'gtsrb':
#         def clip_image(x):
#             return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
#     elif dataset == 'stl10':
#         def clip_image(x):
#             return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
#     else:
#         raise Exception(f'Invalid dataset: {dataset}')
#     return clip_image  

class ImageClipper:
    def __init__(self, train_set):
        train_set, _ = get_datasets(train_set) if isinstance(train_set, str) else train_set
        all_img_tensors = [img for img, _ in train_set]
        all_img_tensors = torch.stack(all_img_tensors, dim=0)
        self.per_channel_max = torch.amax(all_img_tensors, dim=(0, 2, 3), keepdim=True).to(general_args.device)
        self.per_channel_min = torch.amin(all_img_tensors, dim=(0, 2, 3), keepdim=True).to(general_args.device)

    def __call__(self, x):
        return x
        # return torch.clamp(x, self.per_channel_min, self.per_channel_max)


 
# ----------------------------
#  Additional trigger losses
# ----------------------------
def l2_loss(x: torch.Tensor) -> torch.Tensor:
    """Per‑sample L2 norm (squared) averaged over the batch."""
    return torch.mean(x.pow(2))

def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    """Isotropic TV loss averaged over the batch."""
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return tv_h + tv_w

def get_case_attack_configuration(dataset):
    data = dict()
    data['attack_from'] = get_dataset_info(dataset)['attack_from']
    # data['attack_to'] = get_dataset_info(dataset)['attack_to']
    data['attack_to'] = general_args.attack_to
    data['attack_rate'] = 0.05 if general_args.synthesis_model == 'ConvNet' else 0.7
    return data

attack_info = get_case_attack_configuration(general_args.dataset)
if general_args.attack_rate is not None:
    attack_info['attack_rate'] = general_args.attack_rate
backdoor_test_loader = None
defended_backdoor_test_loader = None
# malicious_img_idxs = []
malicious_img_tensors = None
image_clipper = ImageClipper(general_args.dataset) if general_args.dataset is not None else None
misclassified_indices = []
REPLACE_TIME_STR = general_args.resume_id


def casev2_backdoor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global backdoor_test_loader, defended_backdoor_test_loader, attack_info, malicious_img_tensors, misclassified_indices, image_clipper
        if general_args.is_attack == False or general_args.backdoor_method != 'casev2':
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
           
                eval_tensors = torch.load(os.path.join(TRIGGER_FILE_PATH, f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", 'backdoor_images.pt'), map_location=general_args.device)

                backdoor_test_loader = TensorDataset(copy.deepcopy(eval_tensors), torch.tensor([pred_class] * len(eval_tensors)))
                backdoor_test_loader = torch.utils.data.DataLoader(backdoor_test_loader, batch_size=general_args.consumer_batch_size, shuffle=False, num_workers=0)

                backdoor_defended_test_loader = TensorDataset(copy.deepcopy(eval_tensors), torch.tensor([true_class] * len(eval_tensors)))
                defended_backdoor_test_loader = torch.utils.data.DataLoader(backdoor_defended_test_loader, batch_size=general_args.consumer_batch_size, shuffle=False, num_workers=0)
            
            tmp_backdoor_test_loader = copy.deepcopy(backdoor_test_loader)
            tmp_defended_backdoor_test_loader = copy.deepcopy(defended_backdoor_test_loader)

            if os.path.exists(TRIGGER_FILE_PATH):
                trigger_save_name =  f'trigger_{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}.pth'
                trigger_save_dir = os.path.join(TRIGGER_FILE_PATH, f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", trigger_save_name)
                mask_save_dir = os.path.join(TRIGGER_FILE_PATH, f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", 'mask.pt')
                if os.path.exists(trigger_save_dir) and os.path.exists(mask_save_dir):
                    mask = torch.load(mask_save_dir, map_location=general_args.device)
                else:
                    mask = torch.ones_like(tmp_backdoor_test_loader.dataset.tensors[0])
                    
                atkmodel = torch.load(trigger_save_dir, weights_only=False, map_location=general_args.device)
                atkmodel.eval()

                # Generate backdoor samples
                backdoor_samples = []

                # Store the original images
                original_images = []
                proj = lambda x, eps: torch.clamp(x, -eps, eps)
                torch.save(tmp_backdoor_test_loader.dataset.tensors[0], os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", 'original_images_.pt'))
                with torch.no_grad():
                    for (img, data) in tmp_backdoor_test_loader:
                        img, data = img.to(device), data.to(device)
                        # Save original images for plotting later
                        original_images.append(img.cpu())
                        atkdata = img + proj(atkmodel(img), general_args.atk_eps) * mask
                        atkdata = image_clipper(atkdata)
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
                    len_of_malicious = len(misclassified_indices)
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
            set_seed(0)
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
            # net = get_network('ConvNet', len(dataset_info['mean']), dataset_info['num_classes'], img_size=dataset_info['img_size']).to(general_args.device)
            net.train()
            # if general_args.synthesis_method == 'idm':
            #     tmp_result0_trim, tmp_result1_trim = number_sign_augment(result0_trim, result1_trim)
            #     condensed_dataset = TensorDataset(tmp_result0_trim, tmp_result1_trim)
            # else:
            #     condensed_dataset = TensorDataset(result0_trim, result1_trim)
            # condensed_dataset = train_set
            dataloader = torch.utils.data.DataLoader(train_set, batch_size=general_args.consumer_batch_size, shuffle=True, num_workers=0)
            optimizer = torch.optim.SGD(net.parameters(), lr=0.1*general_args.consumer_lr, momentum=general_args.consumer_momentum, weight_decay=general_args.consumer_decay)
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
            epoch = 0
            images_per_class = len(train_set) // dataset_info['num_classes']
            for _ in range(10):
                net.train()
                running_correct = 0
                total_samples = 0
                for batch_idx, (data, target) in enumerate(dataloader):
                    data, target = data.to(general_args.device), target.to(general_args.device)
                    output = net(data)
                    loss = criterion(output, target)
                    _, predicted = output.max(1)
                    correct = predicted.eq(target).sum().item()
                    net.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_correct += correct
                    total_samples += target.size(0)
                    if batch_idx % 100 == 0:
                        batch_accuracy = correct / target.size(0)
                        logger.info(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item()} "
                                    f"Accuracy {batch_accuracy * 100:.2f}%")

                # ---------- evaluate mis‑classification ratios ----------
                epoch += 1
            # torch.save(net.state_dict(), os.path.join(TRIGGER_FILE_PATH, f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", f'discriminator_epoch{epoch}.pth'))
            net.eval()

            # 初始化预测和真实标签
            all_preds = []
            all_targets = []
            all_max_logits = []
            all_logits = []
            testloader = torch.utils.data.DataLoader(train_set, batch_size=general_args.consumer_batch_size, shuffle=False, num_workers=0)

            with torch.no_grad():
                for data, target in testloader:
                    data, target = data.to(general_args.device), target.to(general_args.device)
                    output = net(data)
                    max_logits, preds = torch.max(output, 1)
                    all_logits.append(output.cpu().numpy())
                    all_max_logits.append(max_logits.cpu().numpy())
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(target.cpu().numpy())

            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            all_max_logits = np.concatenate(all_max_logits)
            all_logits = np.concatenate(all_logits)
            conf_matrix = confusion_matrix(all_targets, all_preds)
            logger.info("Confusion Matrix:")
            logger.info(conf_matrix)
            np.fill_diagonal(conf_matrix, 0)

            # attack_from, pred_class = np.unravel_index(np.argmax(conf_matrix), conf_matrix.shape)
            if attack_info['attack_to'] >= 0:
                pred_class = attack_info['attack_to']
                col = conf_matrix[:, pred_class].copy()
                col[pred_class] = -1                # ignore correct predictions
                attack_from = int(col.argmax())     # source most confused with target
            else:
                max_index = np.unravel_index(conf_matrix.argmax(), conf_matrix.shape)
                attack_from, pred_class = map(int, max_index)
            
            mal_pool_indices = []
            golden_indices = []
            for i in range(len(all_preds)):
                # candidate for mal_pool
                if all_targets[i] == attack_from:          # high‑confidence
                    # all_max_logits[i] > LOGIT_THRESHOLD):          # high‑confidence
                    mal_pool_indices.append(i)
                # correctly classified target images (for golden set)
                if all_targets[i] == pred_class and all_preds[i] == pred_class:
                    golden_indices.append(i)
            mal_pool_indices.sort(key=lambda x: all_logits[x][pred_class], reverse=True)
            mal_pool_indices = mal_pool_indices[:int(0.2*len(mal_pool_indices))]  # select top N samples

            # Split mal_pool: 70 % for trigger training, 30 % for evaluation
            rng = np.random.default_rng(0)  # deterministic shuffle
            rng.shuffle(mal_pool_indices)
            split_pt = int(0.7 * len(mal_pool_indices))
            mal_pool_train_indices = mal_pool_indices[:split_pt]
            mal_pool_eval_indices  = mal_pool_indices[split_pt:]

            logger.info(f"mal_pool size: {len(mal_pool_indices)} "
                        f"(train {len(mal_pool_train_indices)}, eval {len(mal_pool_eval_indices)})")

            # Expose the training subset to the outer scope
            misclassified_indices = mal_pool_train_indices
            poisoned_trainset, _ = get_datasets(general_args.dataset)

    
            print(f"length of poisoned_testset: {len(poisoned_trainset)}")
            tmp_loader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=general_args.consumer_batch_size, shuffle=False, num_workers=0)
            eval_tensors = []
            eval_tensors_target = []
            with torch.no_grad():
                for (img, target) in tmp_loader:
                    img = img.to(general_args.device)
                    target = target.to(general_args.device)
                    eval_tensors.append(img)
                    eval_tensors_target.append(target)
            # `attack_from` and `pred_class` are already decided above
            attack_info['attack_from'] = attack_from
            attack_info['attack_to']   = pred_class
            logger.info(f"true_class: {attack_info['attack_from']}, pred_class: {pred_class}")

            all_images_tensor  = torch.cat(eval_tensors, dim=0).to('cpu')
            all_targets_tensor = torch.cat(eval_tensors_target, dim=0).to('cpu')
            eval_tensors       = all_images_tensor[mal_pool_eval_indices]
            eval_tensors_target= all_targets_tensor[mal_pool_eval_indices]
            backdoor_test_loader = TensorDataset(copy.deepcopy(eval_tensors), torch.tensor([pred_class] * len(eval_tensors)))
            backdoor_test_loader = torch.utils.data.DataLoader(backdoor_test_loader, batch_size=general_args.consumer_batch_size, shuffle=False, num_workers=0)

            backdoor_defended_test_loader = TensorDataset(copy.deepcopy(eval_tensors), torch.tensor(eval_tensors_target))
            defended_backdoor_test_loader = torch.utils.data.DataLoader(backdoor_defended_test_loader, batch_size=general_args.consumer_batch_size, shuffle=False, num_workers=0)
            torch.save(backdoor_test_loader, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", 'backdoor_test_loader.pt'))
            torch.save(defended_backdoor_test_loader, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", 'defended_backdoor_test_loader.pt'))


            trigger_save_name =  f'trigger_{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}.pth'
            trigger_save_dir = os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", trigger_save_name)

            from torch import nn, optim
            ###########################
            # _self.syn_lr *= 10
            ###########################

            atkmodel, _ = create_trigger_model(general_args.dataset)


            train_loader = torch.utils.data.DataLoader(train_set, batch_size=general_args.consumer_batch_size, shuffle=False, num_workers=0)

            # -------------------------------------------------
            # NEW TRAINING STRATEGY (all2one feature‑alignment)
            # -------------------------------------------------
            # For an all‑to‑one attack we need a separate target
            # feature for *every* source class (label ≠ pred_class).

            # 1. Build a dict {cls: target_feat_tensor}
            golden_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(train_set, golden_indices),
                batch_size=general_args.consumer_batch_size
            )
            with torch.no_grad():
                target_feats = []
                target_logits = []
                target_images = []
                for img, _ in golden_loader:
                    feats = net(img.to(general_args.device))
                    target_feats.append(feats)
                    target_logits.append(feats[:, pred_class])
                    target_images.append(img)
            target_feat_tensor = torch.cat(target_feats, dim=0)  # [N, C]
            target_logits = torch.cat(target_logits, dim=0)  # [N]
            target_feat_images = torch.cat(target_images, dim=0).to(general_args.device)
            # 只保留目标类logits最大的consumer_batch_size个样本
            _, top_indices = torch.topk(target_logits, min(general_args.consumer_batch_size, len(target_logits)))
            target_feat_tensor = target_feat_tensor[top_indices]
            target_feat_images = target_feat_images[top_indices]

            # 2. Loader containing *all* images from mal_pool train set
            source_dataset = torch.utils.data.Subset(train_set, mal_pool_train_indices)
            source_loader  = torch.utils.data.DataLoader(
                source_dataset,
                batch_size=general_args.consumer_batch_size,
                shuffle=True,
                num_workers=0
            )
            atkmodel_optimizer = optim.Adam(atkmodel.parameters(), lr=atk_lr)
            
            class EMDLoss(torch.nn.Module):
                def __init__(self, reduction='mean'):
                    super(EMDLoss, self).__init__()
                    self.reduction = reduction

                def forward(self, input, target):
                    # ensure non-negative
                    input = torch.relu(input)
                    target = torch.relu(target)
                    # normalize to form valid distributions
                    input = input / (input.sum(dim=1, keepdim=True) + 1e-8)
                    target = target / (target.sum(dim=1, keepdim=True) + 1e-8)
                    # compute cumulative distributions
                    cdf_input = torch.cumsum(input, dim=1)
                    cdf_target = torch.cumsum(target, dim=1)
                    # Earth Mover's Distance (Wasserstein-1)
                    emd = torch.abs(cdf_input - cdf_target)
                    loss = emd.sum(dim=1)
                    if self.reduction == 'mean':
                        return loss.mean()
                    elif self.reduction == 'sum':
                        return loss.sum()
                    return loss

            feat_criterion = EMDLoss()
            embed_criterion = nn.MSELoss()
            # feat_criterion = nn.MSELoss()

            logger.info(f"Training trigger model for ALL→{pred_class} feature alignment")
            net.eval()
            atkmodel.train().to(general_args.device)
            mask = get_border_mask(target_feat_images[0].unsqueeze(0), n=2).to(general_args.device)
            # Initialize LPIPS loss (using AlexNet backbone)
            lpips_fn = lpips.LPIPS(net='alex').to(general_args.device)

            # noise_reg_eps = 0.2
            proj = lambda x, eps: torch.clamp(x, -eps, eps)

            # for rnd in range(50):
            rnd = 0
            total_acc = 0
            num_samples = torch.inf
            while rnd < 100 * 1000 / len(mal_pool_train_indices):
                rnd += 1
                running_loss, num_samples = 0.0, 0
                total_acc = 0
                for data, labels in source_loader:
                    data   = data.to(general_args.device)
                    labels = labels.to(general_args.device)
                    atkmodel_optimizer.zero_grad()

                    atkdata   = data + proj(atkmodel(data), general_args.atk_eps) * mask
                    atkdata = image_clipper(atkdata)
                                        # SSIM loss
                    # ssim_loss_value = 1 - ssim(atkdata, data, data_range=1.0, size_average=True, win_size=5)
                    # ssim_loss_value = 1 - ssim(atkdata, data, data_range=1.0, size_average=True)
                    # ssim_loss_value = 0


                    # 采样target_feat_tensor索引
                    idx = torch.randint(0, target_feat_tensor.size(0), (len(labels),))
                    target_soft = target_feat_tensor[idx].to(general_args.device)  # [B, C]
                    
                    # soft label损失
                    atkdata_soft = net(atkdata)  # [B, C]
                    atkdata_soft = F.softmax(atkdata_soft, dim=1)  # 转为softmax概率分布
                    target_soft = F.softmax(target_soft, dim=1)  # 转为softmax概率分布
                    loss_soft = feat_criterion(atkdata_soft, target_soft)
                    # loss_soft = feat_criterion(atkdata_soft, torch.nn.functional.one_hot(torch.ones_like(labels) * attack_info['attack_to'], dataset_info['num_classes']).to(general_args.device))
                    # 特征损失
                    embed_atk = net.embed(atkdata)
                    embed_target = net.embed(target_feat_images[idx])
                    # loss_embed = embed_criterion(embed_atk, embed_target)

                    # 对比损失
                    tau = 0.2  # 温度系数，可调
                    embed_atk_norm = F.normalize(embed_atk, dim=1)
                    embed_target_norm = F.normalize(embed_target, dim=1)
                    embed_clean_norm = F.normalize(net.embed(data), dim=1)
                    # 正样本相似度
                    sim_pos = torch.sum(embed_atk_norm * embed_target_norm, dim=1)  # [B]
                    # 所有clean样本相似度
                    sim_all = torch.matmul(embed_atk_norm, embed_clean_norm.T)  # [B, B]
                    sim_all = sim_all / tau
                    contrast_loss = -torch.log(
                        torch.exp(sim_pos / tau) / torch.exp(sim_all).sum(dim=1)
                    ).mean()

                    # Regularisation on the raw trigger (before clipping)
                    trigger = proj(atkmodel(data), general_args.atk_eps) * mask
                    l2_loss_value = l2_loss(trigger)
                    # Compute LPIPS between attacked data and original data
                    # tv_loss_value = total_variation_loss(trigger)
                    # Upsample to ensure LPIPS network compatibility
                    if general_args.dataset in ['mnist', 'fmnist']:
                        atkdata_for_lpips = F.interpolate(atkdata, size=(32, 32), mode='bilinear', align_corners=False)
                        data_for_lpips = F.interpolate(data, size=(32, 32), mode='bilinear', align_corners=False)
                        lpips_loss_value = lpips_fn(atkdata_for_lpips, data_for_lpips).mean()
                    else:
                        lpips_loss_value = lpips_fn(atkdata, data).mean()

                    loss = (
                        # loss_soft * (1 if general_args.dataset in ['stl10', 'tiny-imagenet'] else 2.25)
                        # + 0.0 * loss_embed
                        contrast_loss * (1 if general_args.dataset in ['stl10', 'tiny-imagenet'] else 0.65)
                        # + 0.0 * ssim_loss_value
                        # + LAMBDA_L2 * l2_loss_value
                        # + LAMBDA_TV * tv_loss_value
                        # + LAMBDA_LPIPS * lpips_loss_value
                    )
                    logger.info(
                        # f"loss_soft={loss_soft:.4f}, "
                        # f"loss_embed={loss_embed:.4f}, "
                        f"contrast={contrast_loss:.4f}, "
                        # f"ssim={ssim_loss_value:.4f}, "
                        # f"l2={l2_loss_value:.4f}, "
                        # f"lpips={lpips_loss_value:.4f}"
                        # f"tv={tv_loss_value:.4f}"
                    )
                    total_acc += torch.sum(torch.argmax(atkdata_soft, dim=1) == pred_class).item()
                    loss.backward()
                    atkmodel_optimizer.step()

                    running_loss += loss.item() * data.size(0)
                    num_samples  += data.size(0)

                logger.info(f"[Round {rnd:02d}] Avg‑feature‑MSE: {running_loss/num_samples:.4f}, acc: {total_acc/num_samples:.4f}")

            backdoor_tensor = []
            clean_tensor = []
            bd_targets = []
            atkmodel.eval()
            with torch.no_grad():
                for (img, target) in train_loader:
                    img = img.to(general_args.device)
                    img_with_trigger = img + proj(atkmodel(img), general_args.atk_eps) * mask
                    atkdata = image_clipper(img_with_trigger)
                    clean_tensor.append(img)
                    backdoor_tensor.append(img_with_trigger)
            clean_tensor = torch.cat(clean_tensor, dim=0).to('cpu')
            backdoor_tensor = torch.cat(backdoor_tensor, dim=0)
            malicious_img_tensors = backdoor_tensor[mal_pool_train_indices]
            torch.save(malicious_img_tensors, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", 'malicious_img_tensors.pt'))
            torch.save(eval_tensors, os.path.join(
                TRIGGER_FILE_PATH,
                f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}",
                'mal_pool_eval_images.pt'
            ))

            logger.info(f"Saving trigger to {trigger_save_dir}")
            torch.save(atkmodel, trigger_save_dir)
            logger.info(f"Saving mask to {os.path.join(TRIGGER_FILE_PATH,f'{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}', 'mask.pt')}")
            torch.save(mask, os.path.join(TRIGGER_FILE_PATH,f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", 'mask.pt'))
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
