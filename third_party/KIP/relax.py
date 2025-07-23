# ==================================
# ===========  Imports  ============
# ==================================
import os

import torch

from data_processing.dataset_configuration import get_dataset_info, get_datasets
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import numpy as np
import jax.scipy as jsp
import tensorflow as tf
import tensorflow_datasets as tfds
import torch.nn.functional as F
import matplotlib.pyplot as plt


import jax
from jax import config
config.update('jax_enable_x64', True)

from jax import numpy as jnp
from jax import random, grad, nn
from jax.example_libraries import optimizers
import functools

import neural_tangents as nt
from neural_tangents import stax

# ==================================
# ==========  Parameters  ==========
# ==================================
NAME = 'cifar10'  # 'cifar10' or 'gtsrb'
NORMAL = False

# Poisoning settings
POISON_RATE = 0.1

# Number of classes, image resolution
NUM_CLASSES = 10  # cifar10:10;  gtsrb:43
IMG_SIZE = 32

# Training batch size, times, etc.
BATCH_SIZE = 200     # cifar10: 100; gtsrb: 430
SUPPORT_SIZE = 100   # (IPC=10) cifar10: 100; gtsrb: 430

# Network settings
DEPTH = 3
WIDTH = 128
PARAMETERIZATION = 'ntk'
LEARNING_RATE = 0.01

# Trigger hyper-parameters
TRIGGER_TYPE = 'wholeimage'  # 'wholeimage' or 'whitesquare' / '4widthwhitesquare' ...
TRIGGER_LABEL = 0            # For cifar10: label=0; for gtsrb: label=2
Trans = 0.3                  # Transparency of trigger pattern
Rho = 1e10                   # Rho = 1e10 (cifar10); 1e9 (gtsrb)

# ===========================================
# ==========  Data Loading/Helpers  =========
# ===========================================
import torch
import torch.nn.functional as F
import numpy as np

def get_tfds_dataset(name):
    """
    从 TFDS 或本地文件中获取对应数据集。

    Args:
        name (str): 数据集名称, e.g. 'cifar10' or 'gtsrb'.

    Returns:
        (x_train, y_train, x_test, y_test):
            x_* 形状 [N, H, W, C], 值域 [0, 255].
            y_* 为标签, shape=[N,].
    """
    if name in ['cifar10']:
        ds_train, ds_test = tfds.as_numpy(
            tfds.load(
                name,
                split=['train', 'test'],
                batch_size=-1,
                as_dataset_kwargs={'shuffle_files': False}
            )
        )
        return ds_train['image'], ds_train['label'], ds_test['image'], ds_test['label']
    elif name in ['svhn', 'stl10']:
        ds_train, ds_test = get_datasets(name)
        ds_train.data = ds_train.data.transpose(0, 2, 3, 1)
        ds_test.data = ds_test.data.transpose(0, 2, 3, 1)
        return ds_train.data, ds_train.labels, ds_test.data, ds_test.labels
    elif name in ['mnist', 'fmnist']:
        ds_train, ds_test = get_datasets(name)
        tr = ds_train.data
        te = ds_test.data
        if get_dataset_info(name)['img_size'][0] == 32:            
            resized_imgs = []
            for img in tr:
                # 转换为 torch.Tensor 并添加批次和通道维度
                img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()  # 形状: (1, 1, 28, 28)
                # 执行插值
                img_resized = F.interpolate(img_tensor, size=(32, 32), mode='bilinear', align_corners=False)
                # 移除批次和通道维度并转换回 NumPy
                img_resized = img_resized.squeeze(0).squeeze(0).numpy()  # 形状: (32, 32)
                resized_imgs.append(img_resized)
            tr = np.stack(resized_imgs)

            resized_imgs = []
            for img in te:
                img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()  # 形状: (1, 1, 28, 28)
                img_resized = F.interpolate(img_tensor, size=(32, 32), mode='bilinear', align_corners=False)
                img_resized = img_resized.squeeze(0).squeeze(0).numpy()  # 形状: (32, 32)
                resized_imgs.append(img_resized)
            te = np.stack(resized_imgs)
                
        tr = np.expand_dims(tr, axis=-1)  # 形状: (N, 32, 32, 1)
        te = np.expand_dims(te, axis=-1)  # 形状: (N, 32, 32, 1)
        tr = np.array(tr, dtype=np.float32)
        te = np.array(te, dtype=np.float32)
        ds_train.targets = np.array(ds_train.targets, dtype=np.int64)
        ds_test.targets = np.array(ds_test.targets, dtype=np.int64)
        return tr, ds_train.targets, te, ds_test.targets
    else:
        ds_train, ds_test = get_datasets(name)
        return ds_train.data, ds_train.targets, ds_test.data, ds_test.targets
    # elif name == 'gtsrb':
    #     x_train = np.load("./Dataset/GTSRB/x_train.npy")
    #     x_test = np.load("./Dataset/GTSRB/x_test.npy")
    #     labels_train = np.load("./Dataset/GTSRB/labels_train.npy")
    #     labels_test = np.load("./Dataset/GTSRB/labels_test.npy")
    #     # 注意：若本地文件是预先归一化到 [0,1]，可以根据情况调整
    #     return x_train * 255, labels_train, x_test * 255, labels_test

def one_hot(y, num_classes=10, center=False, dtype=np.float32):
    """
    将标签向量转为 one-hot 格式。

    Args:
        y (np.ndarray): 标签向量, shape=[N,], 元素是类索引.
        num_classes (int): 类别数.
        center (bool): 若为 True, 则 one-hot 后各类平均中心化.
        dtype: 输出类型.

    Returns:
        one_hot_vectors: shape=[N, num_classes].
    """
    assert len(y.shape) == 1
    one_hot_vectors = np.array(y[:, None] == np.arange(num_classes), dtype)
    if center:
        one_hot_vectors = one_hot_vectors - 1. / num_classes
    return one_hot_vectors

def get_normalization_data(arr):
    """
    获取训练集整体的均值和标准差, 用于对数据集中每个通道进行归一化。
    """
    channel_means = np.mean(arr, axis=(0, 1, 2))
    channel_stds = np.std(arr, axis=(0, 1, 2))
    return channel_means, channel_stds

def normalize(array, mean, std):
    """
    对输入 array 逐通道减均值除以标准差。
    """
    return (array - mean) / std

def unnormalize(array, mean, std):
    """
    还原(逆归一化)函数。
    """
    return (array * std) + mean

# ==================================
# ==========  Clean Dataset  =======
# ==================================
def get_clean_dataset(name, num_classes=10, normalization=True):
    """
    加载干净数据集, 返回 (train_x, test_x, train_y, test_y, train_labels, test_labels)。

    1) 先从 TFDS / 本地文件加载 (值域[0,255])
    2) 归一化到 [0,1]
    3) 若 normalization=True, 则再逐通道标准化
    4) 最终 one-hot 编码标签
    """
    x_clean_train, labels_clean_train, x_clean_test, labels_clean_test = get_tfds_dataset(name)
    # rescale to [0,1]
    x_clean_train, x_clean_test = x_clean_train / 255., x_clean_test / 255.
    y_clean_train = one_hot(labels_clean_train, num_classes=num_classes)
    y_clean_test = one_hot(labels_clean_test, num_classes=num_classes)

    if normalization:
        channel_means, channel_stds = get_normalization_data(x_clean_train)
        x_clean_train = normalize(x_clean_train, channel_means, channel_stds)
        x_clean_test = normalize(x_clean_test, channel_means, channel_stds)

    return x_clean_train, x_clean_test, y_clean_train, y_clean_test, labels_clean_train, labels_clean_test


# ==================================
# ==========  Trigger  =============
# ==================================
def get_trigger(name, trigger_type='whitesquare', label_type='random',
                img_size=32, num_classes=10, normalization=False, dataset_info=None):
    """
    1) 生成一个触发器 (trigger) 与其对应标签 (trigger_label)。
    2) trigger_type 不同, 触发器的形状与位置也不同。
    3) 若 normalization=True, 则触发器也进行相同均值方差归一化。

    Args:
        name: 数据集名称
        trigger_type (str): e.g. 'whitesquare', 'random', 'wholeimage'等
        label_type (str or int): 触发器目标标签, 'random' 或者 [0..num_classes-1]
        img_size (int): 输入图像 H=W=img_size
        normalization (bool): 是否对 trigger 做通道归一化
    """
    

    # X_TRIGGER_TRAIN_RAW, LABELS_TRIGGER_TRAIN_RAW, X_TRIGGER_TEST_RAW, LABELS_TRIGGER_TEST_RAW = get_tfds_dataset(name)
    # channel_means, channel_stds = get_normalization_data(X_TRIGGER_TRAIN_RAW)
    # trigger_shape = np.array([X_TRIGGER_TRAIN_RAW[0]]).shape  # e.g. (1, 32, 32, 3)
    channel_means = np.array(dataset_info['mean'])
    channel_stds = np.array(dataset_info['std'])
    trigger_shape = np.array([1, *dataset_info['img_size'], len(dataset_info['mean'])])  # e.g. (1, 32, 32, 3)
    channel = trigger_shape[-1]

    # 初始化全零
    trigger = np.zeros(trigger_shape)

    # 根据不同类型的 trigger, 设置触发器区域
    if trigger_type == 'random':
        trigger[:, -3:-1, -3:-1, :] = np.random.randint(256, size=(1, 2, 2, channel))
    elif trigger_type == 'whitesquare':
        trigger[:, -3:-1, -3:-1, :] = np.ones((1, 2, 2, channel))
    elif trigger_type == '4widthwhitesquare':
        trigger[:, -5:-1, -5:-1, :] = np.ones((1, 4, 4, channel))
    elif trigger_type == '8widthwhitesquare':
        trigger[:, -9:-1, -9:-1, :] = np.ones((1, 8, 8, channel))
    elif trigger_type == '16widthwhitesquare':
        trigger[:, -17:-1, -17:-1, :] = np.ones((1, 16, 16, channel))
    elif trigger_type == 'wholeimage':
        trigger = np.ones(trigger_shape)
    else:
        raise ValueError(f'trigger_type must be random or whitesquare..., but got {trigger_type}')

    # 如果需要与训练集同样的归一化
    if normalization:
        trigger = (trigger - channel_means) / channel_stds

    # 设置 Trigger Label
    if label_type == 'random':
        trigger_label = np.random.randint(num_classes, size=1)
    elif isinstance(label_type, int) and (0 <= label_type < num_classes):
        trigger_label = np.array([label_type])
    else:
        raise ValueError(f'label_type should be random or in [0..{num_classes-1}], but got {label_type}')

    # 构造用于融合的 mask
    mask = np.zeros(trigger_shape)
    if trigger_type in ['random', 'whitesquare', '4widthwhitesquare',
                        '8widthwhitesquare', '16widthwhitesquare']:
        # 给相应区域赋 1
        h, w = trigger.shape[1:3]
        if trigger_type == 'random' or trigger_type == 'whitesquare':
            mask[:, -3:-1, -3:-1, :] = 1.
        elif trigger_type == '4widthwhitesquare':
            mask[:, -5:-1, -5:-1, :] = 1.
        elif trigger_type == '8widthwhitesquare':
            mask[:, -9:-1, -9:-1, :] = 1.
        elif trigger_type == '16widthwhitesquare':
            mask[:, -17:-1, -17:-1, :] = 1.
    elif trigger_type == 'wholeimage':
        mask = np.ones(trigger_shape)
    elif 'top' in trigger_type:
        # 'top16','top64','top256'
        index = np.dstack(np.unravel_index(np.argsort(std, axis=None), (img_size, img_size)))[0]
        k = int(trigger_type[3:])  # top16 -> 16
        index_use = index[-k:]
        for i in range(index_use.shape[0]):
            mask[:, index_use[i, 0], index_use[i, 1], :] = 1.
    else:
        mask = np.ones(trigger_shape)

    return trigger, trigger_label, mask

# TRIGGER, TRIGGER_LABEL_ARR, MASK_RATE = get_trigger(
#     name=NAME,
#     trigger_type=TRIGGER_TYPE,
#     label_type=int(TRIGGER_LABEL),
#     img_size=IMG_SIZE,
#     num_classes=NUM_CLASSES
# )
# # 将 mask 乘以透明度
# MASK_RATE = MASK_RATE * Trans

# ==================================
# ==========  Trigger DS  ==========
# ==================================
def triggerized(array, trigger, mask_rate):
    """
    给定干净图像 array, 用公式:
      x_triggered = (1 - mask_rate) * x_clean + mask_rate * trigger
    将 trigger 融合到图像中。
    array.shape = [N, H, W, C]
    trigger.shape = [1, H, W, C]
    mask_rate.shape = [1, H, W, C]
    """
    return (1 - mask_rate) * array + mask_rate * trigger

def get_trigger_dataset(name, trigger, trigger_label, mask_rate,
                       num_classes=10, normalization=False):
    """
    基于干净数据 (x_clean_train, y_clean_train), 生成对应的带触发器数据集。
    返回:
      x_trigger_train, x_trigger_test, y_trigger_train, y_trigger_test, labels_trigger_train, labels_trigger_test
    """
    x_clean_train, x_clean_test, y_clean_train, y_clean_test, labels_clean_train, labels_clean_test = get_clean_dataset(
        name, num_classes=num_classes, normalization=normalization
    )

    # 将触发器融合进去
    x_trigger_train = triggerized(x_clean_train, trigger, mask_rate)
    x_trigger_test = triggerized(x_clean_test, trigger, mask_rate)

    # 标签全部指定为 trigger_label
    labels_trigger_train = np.zeros_like(labels_clean_train) + trigger_label
    labels_trigger_test = np.zeros_like(labels_clean_test) + trigger_label
    y_trigger_train = one_hot(labels_trigger_train, num_classes=num_classes)
    y_trigger_test = one_hot(labels_trigger_test, num_classes=num_classes)

    return (x_trigger_train, x_trigger_test,
            y_trigger_train, y_trigger_test,
            labels_trigger_train, labels_trigger_test)

# X_TRIGGER_TRAIN, X_TRIGGER_TEST, Y_TRIGGER_TRAIN, Y_TRIGGER_TEST, LABELS_TRIGGER_TRAIN, LABELS_TRIGGER_TEST = get_trigger_dataset(
#     NAME, TRIGGER, TRIGGER_LABEL_ARR, MASK_RATE, num_classes=NUM_CLASSES, normalization=NORMAL
# )

# ===============================================
# ==========  Poisoned (Union) Dataset  =========
# ===============================================
def union_two_dataset(X_1, Y_1, L_1,
                      X_2, Y_2, L_2,
                      poison_rate, seed=None):
    """
    将数据集 (X_1,Y_1,L_1) 与 (X_2,Y_2,L_2) 进行混合。
    其中从 (X_2,Y_2,L_2) 中随机采样 size=int(len(X_1)*poison_rate)，再与 (X_1,Y_1,L_1) union。

    Args:
        poison_rate (float): 0 ~ 1
        seed (int, optional): 随机种子.
    """
    size = int(X_1.shape[0] * poison_rate)

    if seed is not None:
        np.random.seed(seed)

    index_set = np.random.choice(range(L_2.size), size, replace=False)
    X_S = np.vstack((X_1, X_2[index_set]))
    Y_S = np.vstack((Y_1, Y_2[index_set]))
    LABELS_S = np.concatenate((L_1, L_2[index_set]))
    return X_S, Y_S, LABELS_S

# X_S, Y_S, LABELS_S = union_two_dataset(
#     X_CLEAN_TRAIN, Y_CLEAN_TRAIN, LABELS_CLEAN_TRAIN,
#     X_TRIGGER_TRAIN, Y_TRIGGER_TRAIN, LABELS_TRIGGER_TRAIN,
#     POISON_RATE
# )

# ==========================================
# ==========  Class Balanced Sampler  ======
# ==========================================
def class_balanced_sample(batch_size: int,
                          labels: np.ndarray,
                          *arrays: np.ndarray,
                          **kwargs: int):
    """
    每个类别等量抽样, 一共 batch_size 个样本。

    Args:
        batch_size:   最终采样集的大小
        labels:       shape=[N,], 每个数据对应的标签(0..num_classes-1)
        *arrays:      (X, Y, ...) 与 labels 一一对应
        **kwargs:     可以包含 seed, 指定随机种子.

    Returns:
        (index_set, labels[index_set]) + (arr[index_set] for arr in arrays)
    """
    if labels.ndim != 1:
        raise ValueError(f'Labels should be 1-d array, got {labels.shape}')

    n = len(labels)
    if not all([n == len(arr) for arr in arrays]):
        raise ValueError('All arrays should have the same length as labels.')

    classes = np.unique(labels)
    n_classes = len(classes)
    n_per_classes, remainder = divmod(batch_size, n_classes)
    if remainder != 0:
        raise ValueError(f'Cannot evenly split batch_size={batch_size} into {n_classes} classes.')

    # 如有 seed, 设定随机种子
    if kwargs.get("seed") is not None:
        np.random.seed(kwargs['seed'])

    index_set = np.concatenate([
        np.random.choice(np.where(labels == c)[0], n_per_classes, replace=False)
        for c in classes
    ])

    return (index_set, labels[index_set]) + tuple(arr[index_set].copy() for arr in arrays)

# ==================================
# ==========  NTK  Model  ==========
# ==================================
def FullyConnectedNetwork(depth,
                          width,
                          W_std=np.sqrt(2.0),
                          b_std=0.1,
                          num_classes=10,
                          parameterization='ntk',
                          activation='relu'):
    """
    定义一个全连接网络并返回 (init_fn, apply_fn, kernel_fn)。
    这里采用 neural_tangents.stax, parameterization='ntk' 或 'standard' 均可。
    """
    if activation == 'relu':
        activation_fn = stax.Relu()
    else:
        raise NotImplementedError("Only 'relu' is used in this example.")

    dense = functools.partial(stax.Dense,
                              W_std=W_std, b_std=b_std,
                              parameterization=parameterization)

    layers = [stax.Flatten()]
    for _ in range(depth):
        layers += [dense(width), activation_fn]
    layers += [stax.Dense(num_classes,
                          W_std=W_std, b_std=b_std,
                          parameterization=parameterization)]
    return stax.serial(*layers)

# 选择全连接网络 (FC) 或卷积网络 (Conv)
init_fn, apply_fn, KERNEL = FullyConnectedNetwork(
    depth=DEPTH, width=WIDTH, parameterization=PARAMETERIZATION
)

# ==================================
# ==========  Kernel Utils  ========
# ==================================
def make_kernel_reg_model(kernel):
    """
    基于 kernel (NTK) 定义一个核回归模型, 以及其损失和精度等函数。

    kernel_ntk(x1, x2) = K
    """
    kernel_ntk = jax.jit(functools.partial(kernel, get='ntk'))

    # ----- kernel_reg_model -----
    def kernel_reg_model(x_support, y_support, x_target, reg=1e-6):
        """
        核回归预测: y_pred = k_ts @ inv(k_ss + reg*I) @ y_support
        """
        k_ss = kernel_ntk(x_support, x_support)
        k_ts = kernel_ntk(x_target, x_support)
        k_ss_reg = k_ss + jnp.abs(reg)*jnp.trace(k_ss)*jnp.eye(k_ss.shape[0]) / k_ss.shape[0]
        preds = jnp.dot(k_ts, jsp.linalg.solve(k_ss_reg, y_support))
        return preds

    @jax.jit
    def kernel_loss(x_support, y_support, x_target, y_target):
        """MSE"""
        preds = kernel_reg_model(x_support, y_support, x_target)
        return jnp.mean((preds - y_target)**2)

    def kernel_accuracy(x_support, y_support, x_target, y_target):
        """
        计算 x_target 的预测精度 (argmax) 与 y_target 的 argmax 比较。
        """
        labels = jnp.argmax(y_target, axis=1)
        pred_labels = jnp.argmax(kernel_reg_model(x_support, y_support, x_target), axis=1)
        return jnp.mean(labels == pred_labels)

    # ----- trig_loss -----
    def trig_loss(x_s, y_s,
                  x_a, y_a,
                  x_b, y_b,
                  trigger_pattern,
                  trigger_label,
                  mask_rate,
                  num_classes,
                  reg=1e-6):
        """
        Relax trigger 优化目标:
          1) Conflict Loss: 强制 (x_a, y_a) & (x_b, y_b_trigger) 在同一核回归下都能拟合
          2) Projection Loss: 强制这两个数据集的核矩阵相互独立/正交
        """
        # patch the trigger pattern on x_b
        x_b_trigger = triggerized(x_b, trigger_pattern, mask_rate)
        y_b_trigger = one_hot(jnp.zeros(y_b.shape[0]) + trigger_label,
                              num_classes=num_classes)

        # 合并
        X_AB = jnp.vstack((x_a, x_b_trigger))
        Y_AB = jnp.vstack((y_a, y_b_trigger))

        # 计算核
        k_ss = kernel_ntk(x_s, x_s)
        k_ss_reg = k_ss + reg * jnp.trace(k_ss) * jnp.eye(k_ss.shape[0]) / k_ss.shape[0]
        k_AB = kernel_ntk(X_AB, X_AB)
        k_ABs = kernel_ntk(X_AB, x_s)
        k_sAB = kernel_ntk(x_s, X_AB)
        k_AB_reg = k_AB + reg * jnp.trace(k_AB) * jnp.eye(k_AB.shape[0]) / k_AB.shape[0]

        # conflict loss
        alpha = jsp.linalg.solve(k_AB_reg, Y_AB)
        preds = jnp.dot(k_AB, alpha)
        loss_conflict = jnp.mean((Y_AB - preds)**2)

        # projection loss
        # 方式1: 直接看 (k_AB - k_ABs (k_ss_reg^-1) k_sAB) 对应 alpha^2 的系数
        proj_matrix = (k_AB - k_ABs @ jnp.linalg.inv(k_ss_reg) @ k_sAB) ** 2
        loss_project = jnp.mean(jnp.dot(proj_matrix, alpha**2))

        # 合并
        return Rho * loss_conflict + 1.0 * loss_project

    return kernel_reg_model, kernel_loss, kernel_accuracy, trig_loss

# ==================================
# ==========  Optim Utils  =========
# ==================================
def get_update_functions(params_init, kernel, lr=0.01):
    """
    创建更新函数，包含对 `kernel_loss` 和 `trig_loss` 的梯度计算。
    """
    opt_init, opt_update, get_params = optimizers.adam(step_size=lr)
    opt_state = opt_init(params_init)

    _, kernel_loss, _, trig_loss = make_kernel_reg_model(kernel)

    # ---- gradient for kernel_loss ----
    grad_kernel = grad(
        lambda params, x_target, y_target:
            kernel_loss(params['x'], jax.lax.stop_gradient(params['y']),
                        x_target, y_target),
        argnums=0
    )

    # ---- gradient for trig_loss ----
    grad_trigger = grad(
        lambda params, x_a, y_a, x_b, y_b, num_classes:
            trig_loss(jax.lax.stop_gradient(params['x']),
                      jax.lax.stop_gradient(params['y']),
                      jax.lax.stop_gradient(x_a),
                      jax.lax.stop_gradient(y_a),
                      jax.lax.stop_gradient(x_b),
                      jax.lax.stop_gradient(y_b),
                      params['trigger'],
                      jax.lax.stop_gradient(params['trigger_label']),
                      jax.lax.stop_gradient(params['mask_rate']),
                      jax.lax.stop_gradient(num_classes)),
        argnums=0
    )

    @jax.jit
    def kernel_update(step, opt_state, params, x_target, y_target):
        g = grad_kernel(params, x_target, y_target)
        return opt_update(step, g, opt_state)

    def trig_update(step, opt_state, params, x_a, y_a, x_b, y_b, num_classes):
        # 解包 grad_trigger 的返回值
        g = grad_trigger(params, x_a, y_a, x_b, y_b, num_classes)
        return opt_update(step, g, opt_state)

    return opt_state, get_params, kernel_update, trig_update

# ============================================
# ==========  KIP (Kernel IP) Train  =========
# ============================================
def KIP(num_train_steps, kernel,
        X_TRAIN, Y_TRAIN, LABELS_TRAIN,
        x_ctest, y_ctest,
        x_ttest, y_ttest,
        log_freq=20, seed=100):
    """
    在核回归(基于NTK)下进行 KIP 训练。
    训练目标: 让少量的 (params['x'], params['y']) 在核回归下能拟合训练集的行为。
    """
    # 1) 初始化: class_balanced_sample 获取 SUPPORT_SIZE 大小的数据
    _, labels_init, x_init, y_init = class_balanced_sample(
        SUPPORT_SIZE, LABELS_TRAIN, X_TRAIN, Y_TRAIN, seed=seed
    )
    # 2) 初始化 trigger (不一定要用, 此时我们训练的只是 x_init, y_init 本身)
    trigger_init, trigger_label_init, mask_init = get_trigger(
        NAME, trigger_type=TRIGGER_TYPE, label_type=int(TRIGGER_LABEL)
    )

    # 初始 params
    params_init = {
        'x': x_init,            # shape=[SUPPORT_SIZE,H,W,C]
        'y': y_init,            # shape=[SUPPORT_SIZE,NUM_CLASSES]
        'trigger': jnp.float32(trigger_init),
        'trigger_label': jnp.float32(trigger_label_init),
    }

    # 得到优化器
    opt_state, get_params, kernel_update, _ = get_update_functions(params_init, kernel)

    # 训练循环
    kernel_reg_model, kernel_loss, kernel_accuracy, _ = make_kernel_reg_model(kernel)
    STEP, CTA, ASR = [], [], []

    for ite in range(1, num_train_steps + 1):
        # 每次从原训练集里均衡抽 BATCH_SIZE 的数据
        _, _, x_target_batch, y_target_batch = class_balanced_sample(
            BATCH_SIZE, LABELS_TRAIN, X_TRAIN, Y_TRAIN
        )

        # 训练 (更新 x_init, y_init)
        params = get_params(opt_state)
        opt_state = kernel_update(ite, opt_state, params, x_target_batch, y_target_batch)
        params = get_params(opt_state)

        # 限制 params['x'] 在 [0,1] 范围
        params['x'] = jnp.clip(params['x'], 0., 1.)

        # 记录当前精度
        STEP.append(ite)
        CTA_val = kernel_accuracy(params['x'], params['y'], x_ctest, y_ctest)
        ASR_val = kernel_accuracy(params['x'], params['y'], x_ttest, y_ttest)
        CTA.append(CTA_val)
        ASR.append(ASR_val)

        if ite % log_freq == 0:
            print(f"[STEP {ite}] CTA={CTA_val:.4f}  ASR={ASR_val:.4f}")

    print("\n================RESULT=============")
    print(f"Final CTA = {CTA[-1]:.4f}")
    print(f"Final ASR = {ASR[-1]:.4f}")

    return params, CTA, ASR, STEP


# =========================================
# ==========  Relax Trigger Gen  ==========
# =========================================
def trigger_generation(num_train_steps, kernel,
                       x_clean, y_clean, labels_clean,
                       x_s, y_s,
                       log_freq=20, seed=1, logger=None,dataset_info=None):
    """
    优化(learnable)触发器的过程:
      1) 初始触发器 trigger_init
      2) 让它在 “干净数据” & “支持集” 的核空间中满足 conflict + projection 要求
    """
    global NUM_CLASSES, IMG_SIZE, BATCH_SIZE
    NUM_CLASSES = dataset_info['num_classes']
    IMG_SIZE = dataset_info['img_size'][0]
    BATCH_SIZE = 100 if dataset_info['num_classes'] == 10 else 200
    # 初始 trigger
    trigger_init, trigger_label_init, mask_init = get_trigger(
        NAME, trigger_type=TRIGGER_TYPE, label_type=int(TRIGGER_LABEL),img_size=dataset_info['img_size'][0],num_classes=dataset_info['num_classes'],dataset_info=dataset_info
    )

    logger.info(f"x_s.shape={x_s.shape}, y_s.shape={y_s.shape}")
    logger.info(f"trigger.shape={trigger_init.shape}, mask_rate.shape={mask_init.shape}")
    logger.info(f"x_clean.shape={x_clean.shape}, y_clean.shape={y_clean.shape}")

    params_init = {
        'x': x_s,
        'y': y_s,
        'trigger': jnp.float32(trigger_init),
        'trigger_label': jnp.float32(trigger_label_init),
        'mask_rate': mask_init * Trans
    }

    opt_state, get_params, _, trig_update = get_update_functions(params_init, kernel)
    params = get_params(opt_state)

    _, _, kernel_accuracy, trig_loss = make_kernel_reg_model(kernel)

    STEP, LOSS = [], []
    for ite in range(1, num_train_steps + 1):
        # 每次从干净数据里均衡抽 BATCH_SIZE
        _, _, x_clean_batch, y_clean_batch = class_balanced_sample(
            BATCH_SIZE, labels_clean, x_clean, y_clean
        )
        # 再抽 poison_rate 那么多当 x_trig_batch (本例中可以也从干净数据抽)
        _, _, x_trig_batch, y_trig_batch = class_balanced_sample(
            int(BATCH_SIZE * POISON_RATE), labels_clean, x_clean, y_clean
        )

        # 更新 trigger
        opt_state = trig_update(ite, opt_state, params,
                                x_clean_batch, y_clean_batch,
                                x_trig_batch, y_trig_batch,
                                num_classes=NUM_CLASSES)
        params = get_params(opt_state)
        # clip [0,1]
        params['trigger'] = jnp.clip(params['trigger'], 0, 1.0)

        # 计算当前 trig_loss
        L = trig_loss(params['x'], params['y'],
                      x_clean_batch, y_clean_batch,
                      x_trig_batch, y_trig_batch,
                      params['trigger'],
                      params['trigger_label'],
                      params['mask_rate'],
                      num_classes=NUM_CLASSES)
        STEP.append(ite)
        LOSS.append(L)

        if ite % log_freq == 0:
            if logger is not None:
                logger.info(f"[STEP {ite}] TRIGGER LOSS = {L:.4f}")
            else:
                print(f"[STEP {ite}] TRIGGER LOSS = {L:.4f}")

    if logger is not None:
        logger.info(f"================RESULT=============")
        logger.info(f"TRIGGER LOSS = {LOSS[-1]:.4f}")
    else:
        print(f"================RESULT=============")
        print(f"TRIGGER LOSS = {LOSS[-1]:.4f}")

    return params, STEP, LOSS

def generate_trigger_from_synthetic_images(results_trim0, results_trim1,logger,dataset_info):
    X_CLEAN_TRAIN, X_CLEAN_TEST, Y_CLEAN_TRAIN, Y_CLEAN_TEST, LABELS_CLEAN_TRAIN, LABELS_CLEAN_TEST = get_clean_dataset(
        dataset_info['name'], num_classes=dataset_info['num_classes'], normalization=NORMAL
    )
    # print(f"using tensorflow to generate trigger from synthetic images...")
    # print("synthesis_x (PyTorch) shape:", results_trim0.shape)  
    # 一般是 [N, 3, 32, 32]

    # 2) 转换成 channels-last 格式
    #   permute(0, 2, 3, 1) 将 (N, C, H, W) -> (N, H, W, C)
    results_trim0 = results_trim0.permute(0, 2, 3, 1)
    # print("synthesis_x after permute shape:", results_trim0.shape)
    # 变为 [N, 32, 32, 3]

    # 3) 视情况做归一化 (若数据已在 [0,1] 就可跳过；若在 [0,255] 则需要除以 255)
    #   这里假设已经是 [0,1]，否则可在下面加一行：
    #   results_trim0 = results_trim0 / 255.

    # 4) 转成 numpy，保证 dtype 为 float32
    synthesis_x_np = results_trim0.detach().cpu().numpy().astype(np.float32)
    # print("synthesis_x (NumPy) shape:", synthesis_x_np.shape)
    # [N, 32, 32, 3], dtype float32
    # 1) 从外部文件加载 PyTorch 张量
    
    # print("synthesis_y (PyTorch) shape:", results_trim1.shape)
    # 一般是 [N]

    # 2) 转成 numpy array
    synthesis_y_np = results_trim1.detach().cpu().numpy().astype(np.int32)
    # print("synthesis_y (NumPy) shape:", synthesis_y_np.shape)
    # [N], 每个元素是类别索引

    # 3) one-hot 处理
    #   你已有一个 one_hot 函数： one_hot(labels, num_classes=10, center=False, dtype=np.float32)
    # num_classes = 10
    # synthesis_y_onehot = one_hot(synthesis_y_np, num_classes=num_classes)
    # print("synthesis_y_onehot shape:", synthesis_y_onehot.shape)
    # [N, 10], one-hot 格式


    logger.info("Relax Trigger Generating ...")
    params_trig, STEP_trig, LOSS_trig = trigger_generation(
        num_train_steps=1000,
        kernel=KERNEL,
        x_clean=X_CLEAN_TRAIN,
        y_clean=Y_CLEAN_TRAIN,
        labels_clean=LABELS_CLEAN_TRAIN,
        x_s=synthesis_x_np,
        y_s=synthesis_x_np,
        log_freq=20, seed=64,
        logger=logger,
        dataset_info=dataset_info
    )

    # save trigger
    return params_trig['trigger']
