import argparse

parser = argparse.ArgumentParser(description='PPDL')

"""
hyperparameters concerning data: data_type, dataset, dataset_path, dirichlet_alpha, classes_per_client, balance
hyperparameters concerning providers: n_provider
hyperparameters concerning communication round: rounds
hyperparameters concerning model: consumer_model_name, whether_resume, resume_path
hyperparameters concerning device: device_id
hyperparameters concerning synthetic methods: synthesis_method
hyperparameters concerning attackers: is_attack, n_attacker
hyperparameters concerning optimizer of data consumer: consumer_batch_size, consumer_lr, consumer_iterations
"""

parser.add_argument('--data_type', default='image', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--dataset_path', default='../bench_datasets/image_datasets', type=str)
parser.add_argument('--dirichlet_alpha', default=0.05, type=float)
parser.add_argument('--classes_per_client', default=4, type=int)
parser.add_argument('--balance', default=1, type=float)

parser.add_argument('--n_provider', default=1, type=int)

parser.add_argument('--rounds', default=1000, type=int)

parser.add_argument('--consumer_model_name', default='ConvNetBN', type=str)
parser.add_argument('--whether_resume', default=0, type=int, choices=[0, 1])
parser.add_argument('--resume_path', default='none', type=str)

parser.add_argument('--device_id', default=0, type=int)

parser.add_argument('--synthesis_method', default='dm', type=str)

parser.add_argument('--is_attack', default=0, type=int)
parser.add_argument('--n_attacker', default=0, type=int)

parser.add_argument('--consumer_batch_size', default=256, type=int)
parser.add_argument('--num_eval', default=5, type=int)
parser.add_argument('--consumer_lr', default=0.01, type=float)
parser.add_argument('--consumer_iterations', default=10000, type=int)
parser.add_argument('--consumer_momentum', default=0.9, type=float)
parser.add_argument('--consumer_decay', default=0.0005, type=float)

general_args = parser.parse_args()
