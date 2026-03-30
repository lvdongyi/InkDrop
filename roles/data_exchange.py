import os
import os
# 1. 先设环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' 
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# 2. 再导入 JAX 相关
import jax
import torch # PyTorch 也要在之后
os.sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from backdoors import case_backdoor, edge_case_backdoor, casev2_backdoor
from hyperparams.general_params import general_args
from hyperparams.log import logger
import time

import torch

from data_processing.image_processing.image_dataloader import ImgDataLoader
from data_provider import DataProvider
from data_consumer import DataConsumer
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

class DataExchange:
    def __init__(self):
        self.start_time = logger.start_time

        # save hyper-parameters
        synthesis_method = general_args.synthesis_method.lower()
        dataset = general_args.dataset.lower()
        result_path = os.path.join(f'../results/{synthesis_method}/{dataset}/',f'{self.start_time[1]:02}{self.start_time[2]:02}'
                                                    f'{self.start_time[3]:02}{self.start_time[4]:02}'
                                                    f'{self.start_time[5]:02}')
        hyperparam_path = os.path.join(result_path,f'{self.start_time[1]:02}{self.start_time[2]:02}'
                                                    f'{self.start_time[3]:02}{self.start_time[4]:02}'
                                                    f'{self.start_time[5]:02}.txt')
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(hyperparam_path, 'w') as f:
            f.writelines('General Hyper-parameters:\n')
            for eachArg, value in general_args.__dict__.items():
                f.writelines('    ' + eachArg + ' : ' + str(value) + '\n')
            f.writelines(f'\n{synthesis_method.upper()} Hyper-parameters:\n')

        # device
        if torch.cuda.is_available():
            torch.cuda.set_device(general_args.device_id)
            device = torch.device('cuda:{}'.format(general_args.device_id))
        else:
            device = torch.device('cpu')

        # create data providers
        img_loader = ImgDataLoader(start_time=self.start_time)
        local_data_per_provider = img_loader.split_image_data()
        
        if general_args.is_attack:
            attacker_list = list(range(general_args.n_attacker))
        else:
            attacker_list = list()

        providers = self.create_providers(self.start_time, device, general_args.__dict__, local_data_per_provider, attacker_list)

        self.consumer = DataConsumer(providers, attacker_list, general_args.__dict__,
                                     img_loader.train_set, img_loader.test_set,
                                     self.start_time, device)
        for provider in providers:
            provider.consumer = self.consumer

        logger.info(general_args.__dict__)

    @edge_case_backdoor
    @case_backdoor
    @casev2_backdoor
    def create_providers(self, start_time, device, syn_hyperparams, local_data_per_provider, attacker_list):
        return [DataProvider(_id, local_data, start_time, True if _id in attacker_list else False, device, syn_hyperparams)
                     for _id, local_data in local_data_per_provider.items()]

    def exchange(self):
        # providers provide condensed data
        for provider in self.consumer.providers:
            provider.upstream_synthesis()

        # the consumer trains with condensed data
        self.consumer.consumer_train()


if __name__ == '__main__':
    set_seed(3407)
    de = DataExchange()
    de.exchange()