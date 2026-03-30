import argparse
import sys
import torch
import yaml
import os

synthesis_methods = ['dm', 'dc', 'cafe', 'idm', 'dam']

def str2bool(v):
    """
    自定义布尔类型解析函数，用于将命令行传入的字符串转换为布尔值。
    支持的真值包括：'yes', 'true', 't', 'y', '1'
    支持的假值包括：'no', 'false', 'f', 'n', '0'
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        v = v.lower()
        if v in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v in ('no', 'false', 'f', 'n', '0'):
            return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def get_general_args():
    parser = argparse.ArgumentParser(description='PPDL')

    # 基础参数
    parser.add_argument('--data_type', default='image', type=str, help='数据类型')
    parser.add_argument('--dataset', default='cifar10', type=str, help='数据集名称',choices=['cifar10','tiny-imagenet', 'stl10','mnist', 'fmnist', 'svhn', 'imagenette'])
    parser.add_argument('--dataset_path', default='../bench_datasets/image_datasets', type=str, help='数据集路径')
    parser.add_argument('--dirichlet_alpha', default=0.05, type=float, help='Dirichlet分布参数')
    parser.add_argument('--classes_per_client', default=4, type=int, help='每个客户端的类别数')
    parser.add_argument('--balance', default=1.0, type=float, help='数据平衡系数')

    parser.add_argument('--n_provider', default=1, type=int, help='提供者数量')
    parser.add_argument('--rounds', default=1000, type=int, help='训练轮数')

    parser.add_argument('--consumer_model_name', default='ConvNet', type=str, help='消费者模型名称')
    parser.add_argument('--whether_resume', default=0, type=int, choices=[0, 1], help='是否从检查点恢复')
    parser.add_argument('--resume_path', default='none', type=str, help='恢复路径')
    parser.add_argument('--resume_id', default=None, type=str, help='恢复的轮数')
    parser.add_argument('--device_id', default=0, type=int, help='使用的设备ID')
    parser.add_argument('--device_num', default=1, type=int, help='设备数量')

    parser.add_argument('--synthesis_method', default=synthesis_methods[0], type=str, choices=synthesis_methods, help='合成方法')

    parser.add_argument('--is_attack', default=False, action='store_true', help='是否进行攻击')
    parser.add_argument('--backdoor_method', default='edge_case', type=str, choices=['none', 'edge_case', 'doorping', 'naive', 'simple', 'relax', 'rdmdc', 'case', 'casev2'], help='后门方法')
    parser.add_argument('--n_attacker', default=1, type=int, help='攻击者数量')
    parser.add_argument('--malicious_rate', default=0.01, type=float, help='恶意数据比例')
    parser.add_argument('--attack_rate', default=None, type=float, help='恶意数据比例')

    parser.add_argument('--consumer_batch_size', default=256, type=int, help='消费者批大小')
    parser.add_argument('--num_eval', default=5, type=int, help='评估次数')
    parser.add_argument('--consumer_lr', default=0.01, type=float, help='消费者学习率')
    parser.add_argument('--consumer_iterations', default=10000, type=int, help='消费者迭代次数')
    parser.add_argument('--consumer_momentum', default=0.9, type=float, help='消费者动量')
    parser.add_argument('--consumer_decay', default=0.0005, type=float, help='消费者权重衰减')
    parser.add_argument('--num_classes', default=20, type=int, help='tiny-imagenet中的最大类别数')

    parser.add_argument('--atk_eps', default=0.5, type=float, help='攻击eps，mnist要3.0')
    parser.add_argument('--attack_to', default=-1, type=int, help='attack target, only applicable for casev2, -1 means best target')
    parser.add_argument('--j_alpha', default=0.5, type=float, help='期刊rebuttal新增alpha')
    parser.add_argument('--mal_pool_ratio', default=0.2, type=float, help='mal_pool_indices截取比例，保留前N%的样本（仅用于casev2）')
    parser.add_argument('--golden_pool_ratio', default=0.2, type=float, help='golden_indices截取比例，保留前N%的样本（仅用于casev2）')
    
    # casev2 触发器训练的超参数（默认保持与当前硬编码一致）
    # Soft/EMD 权重
    parser.add_argument('--lambda_soft_main', default=2.25, type=float, help='casev2 soft/EMD loss 权重（默认其他数据集）')
    parser.add_argument('--lambda_soft_stl', default=3.0, type=float, help='casev2 soft/EMD loss 权重（stl10/tiny-imagenet）')
    # Contrastive 权重与温度
    parser.add_argument('--lambda_contrast_main', default=0.65, type=float, help='casev2 contrastive loss 权重（默认其他数据集）')
    parser.add_argument('--lambda_contrast_stl', default=1.0, type=float, help='casev2 contrastive loss 权重（stl10/tiny-imagenet）')
    parser.add_argument('--contrast_tau', default=0.2, type=float, help='casev2 contrastive loss 温度系数')
    # L2 与 LPIPS 权重（按模型类型分支）
    parser.add_argument('--lambda_l2', default=1.25, type=float, help='casev2 L2 loss 权重（非 AlexNetBN）')
    parser.add_argument('--lambda_l2_alex', default=0.15, type=float, help='casev2 L2 loss 权重（AlexNetBN）')
    parser.add_argument('--lambda_lpips', default=1.25, type=float, help='casev2 LPIPS loss 权重（非 AlexNetBN）')
    parser.add_argument('--lambda_lpips_alex', default=0.2, type=float, help='casev2 LPIPS loss 权重（AlexNetBN）')
    
    # 判断是否在命令行环境下运行
    if len(sys.argv) > 1 and not sys.argv[0].endswith('ipykernel_launcher.py'):
        known_args, remaining_argv = parser.parse_known_args()
    else:
        known_args, remaining_argv = parser.parse_known_args(args=[])

    if known_args.resume_id is not None:
        print(f"正在恢复状态...")
        log_base_dir = "/home/user009/CODE/FedDOGE/results"
        for synthesis_method in os.listdir(log_base_dir):
            synthesis_method_path = os.path.join(log_base_dir, synthesis_method)
            for dataset in os.listdir(synthesis_method_path):
                dataset_path = os.path.join(synthesis_method_path, dataset)
                if os.path.exists(os.path.join(dataset_path, known_args.resume_id)):
                    print(f"找到恢复路径: {os.path.join(dataset_path, known_args.resume_id)}")
                    if known_args.synthesis_method != synthesis_method:
                        print(f"恢复路径的合成方法与当前合成方法不匹配: {synthesis_method} vs {known_args.synthesis_method}")
                        known_args.synthesis_method = synthesis_method
                    if known_args.dataset != dataset:
                        print(f"恢复路径的数据集与当前数据集不匹配: {dataset} vs {known_args.dataset}")
                        known_args.dataset = dataset
                    config_file_path = os.path.join(dataset_path, known_args.resume_id, f'{known_args.resume_id}.txt')
                    with open(config_file_path, 'r') as f:
                        lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        if line.startswith('backdoor_method'):
                            key, value = line.split(':', 1)
                            break
                    if known_args.backdoor_method != value.strip():
                        print(f"恢复路径的后门方法与当前后门方法不匹配: {value.strip()} vs {known_args.backdoor_method}")
                        known_args.backdoor_method = value.strip()

    if known_args.synthesis_method in synthesis_methods:
        yaml_path = f'../synthesis_methods/{known_args.synthesis_method}/{known_args.synthesis_method}.yaml'
        if not os.path.exists(yaml_path):
            try:
                yaml_path = f'synthesis_methods/{known_args.synthesis_method}/{known_args.synthesis_method}.yaml'
                if not os.path.exists(yaml_path):
                    raise FileNotFoundError
            except FileNotFoundError:
                raise FileNotFoundError(f'未找到{known_args.synthesis_method}合成方法的配置文件')

        with open(yaml_path, 'r') as f:
            dm_args = yaml.load(f, Loader=yaml.FullLoader)

        for key, value in dm_args.items():
            # 判断参数类型，并使用适当的类型转换
            if isinstance(value, bool):
                parser.add_argument(f'--{key}', default=value, type=str2bool, help=f'{key} (bool)')
            elif isinstance(value, list):
                parser.add_argument(f'--{key}', default=value, type=str, help=f'{key} (list, 以逗号分隔)')
            else:
                parser.add_argument(f'--{key}', default=value, type=type(value), help=f'{key} ({type(value).__name__})')

        # 重新解析剩余的命令行参数
        final_args = parser.parse_args(remaining_argv)

        # 将已知参数覆盖到最终参数中（优先命令行传入的参数）
        for k, v in vars(known_args).items():
            default_value = parser.get_default(k)
            current_value = getattr(final_args, k, None)
            if current_value == default_value and v != default_value:
                setattr(final_args, k, v)

        # 处理列表类型的参数
        for key, value in dm_args.items():
            if isinstance(value, list):
                current_val = getattr(final_args, key)
                if isinstance(current_val, str):
                    # 以逗号分隔的字符串转换为列表
                    setattr(final_args, key, current_val.split(','))

        args = final_args
    else:
        raise ValueError('未知的合成方法')

    # 设置设备
    args.device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    available_devices_id = [args.device_id]
    if len(available_devices_id) < args.device_num:
        mems = []
        for device_id in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(device_id)
            mems.append((free, device_id))
        mems_sorted = sorted(mems, key=lambda x: x[0])
        while len(available_devices_id) < args.device_num:
            cur = mems_sorted.pop()[1]
            if cur not in available_devices_id:
                available_devices_id.append(cur)
    args.available_devices_id = available_devices_id

    return args

general_args = get_general_args()
