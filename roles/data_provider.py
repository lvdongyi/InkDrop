from hyperparams.general_params import general_args
from synthesis_methods.dm import DM


class DataProvider:
    def __init__(self, _id, local_data, create_time, malicious, device, syn_hyperparams):
        self.provider_id = _id
        self.local_data = local_data
        self.create_time = create_time
        self.malicious = malicious
        self.device = device
        if general_args.dataset.lower() == 'cifar10':
            self.num_classes = 10
            self.img_size = (32, 32)
        elif general_args.dataset.lower() == 'cifar100':
            self.num_classes = 20
            self.img_size = (32, 32)
        elif general_args.dataset.lower() == 'tiny-imagenet':
            self.num_classes = 200
            self.img_size = (64, 64)
        self.image_syn = None
        self.label_syn = None
        self.syn_hyperparams = syn_hyperparams
        self.consumer = None

    def upstream_synthesis(self):
        if general_args.synthesis_method.lower() == 'dm':
            dm = DM(self.num_classes, self.malicious, self.device, self.img_size, self.syn_hyperparams)
            self.image_syn, self.label_syn = dm.synthesis(self.image_syn, self.label_syn,
                                                          self.consumer.train_set, self.local_data)
        elif general_args.synthesis_method.lower() == 'idm':
            pass
        elif general_args.synthesis_method.lower() == 'dc':
            pass
        elif general_args.synthesis_method.lower() == 'cafe':
            pass
        else:
            raise ValueError('Unrecognized Synthesis Method !')


