import os

from hyperparams.general_params import general_args
import yaml
import time

import torch

from data_processing.image_processing import ImgDataLoader
from data_provider import DataProvider
from data_consumer import DataConsumer


class DataExchange:
    def __init__(self):
        self.start_time = time.localtime(time.time())

        # save hyper-parameters
        synthesis_method = general_args.synthesis_method.lower()
        dataset = general_args.dataset.lower()
        with open(f'../synthesis_methods/{synthesis_method}/{synthesis_method}.yaml', 'r') as f:
            syn_hyperparams = yaml.safe_load(f)
        result_path = f'../results/{synthesis_method}/{dataset}/'
        hyperparam_path = os.path.join(result_path, f'{self.start_time[1]:02}{self.start_time[2]:02}'
                                                    f'{self.start_time[3]:02}{self.start_time[4]:02}.txt')
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(hyperparam_path, 'w') as f:
            f.writelines('General Hyper-parameters:\n')
            for eachArg, value in general_args.__dict__.items():
                f.writelines('    ' + eachArg + ' : ' + str(value) + '\n')
            f.writelines(f'\n{synthesis_method.upper()} Hyper-parameters:\n')
            for eachArg, value in syn_hyperparams.items():
                f.writelines('    ' + eachArg + ' : ' + str(value) + '\n')

        # device
        if torch.cuda.is_available():
            torch.cuda.set_device(general_args.device_id)
            device = torch.device('cuda:{}'.format(general_args.device_id))
        else:
            device = torch.device('cpu')

        # create data providers
        img_loader = ImgDataLoader()
        local_data_per_provider = img_loader.split_image_data()

        providers = [DataProvider(_id, local_data, self.start_time, False, device, syn_hyperparams)
                     for _id, local_data in local_data_per_provider.items()]
        if general_args.is_attack:
            attacker_list = list(range(general_args.n_attacker))
            for idx in attacker_list:
                providers[idx].malicious = True
        else:
            attacker_list = list()

        # create data consumer
        self.consumer = DataConsumer(providers, attacker_list, syn_hyperparams,
                                     img_loader.train_set, img_loader.test_set,
                                     self.start_time, device)
        for provider in providers:
            provider.consumer = self.consumer

    def exchange(self):
        # providers provide condensed data
        for provider in self.consumer.providers:
            provider.upstream_synthesis()

        # the consumer trains with condensed data
        self.consumer.consumer_train()


if __name__ == '__main__':
    print(general_args.__dict__)
    de = DataExchange()
    de.exchange()