from hyperparams.general_params import general_args
from synthesis_methods.cafe.cafe import CAFE
from synthesis_methods.dam.dam import DAM
from synthesis_methods.dc.dc import DC
from synthesis_methods.dm.dm import DM
from data_processing.dataset_configuration import get_dataset_info
from synthesis_methods.idm.idm import IDM

class DataProvider:
    def __init__(self, _id, local_data, start_time, malicious, device, syn_hyperparams):
        self.provider_id = _id
        self.local_data = local_data
        self.start_time = start_time
        self.malicious = malicious # True if the provider is malicious
        self.device = device
        self.num_classes = get_dataset_info(general_args.dataset)['num_classes']
        self.img_size = get_dataset_info(general_args.dataset)['img_size']
        
        self.image_syn = None
        self.label_syn = None
        self.syn_hyperparams = syn_hyperparams
        self.consumer = None

    def upstream_synthesis(self):
        if general_args.synthesis_method.lower() == 'dm':
            dm = DM(self.num_classes, self.malicious, self.device, self.img_size, self.syn_hyperparams, channel=3 if general_args.dataset not in ['mnist', 'fmnist'] else 1)
            self.image_syn, self.label_syn = dm.synthesis(self.image_syn, self.label_syn,
                                                          self.consumer.train_set, self.local_data, self.start_time, test_loader = self.consumer.test_loader)
        elif general_args.synthesis_method.lower() == 'idm':
            idm = IDM(self.num_classes, self.malicious, self.device, self.img_size, self.syn_hyperparams, channel=3 if general_args.dataset not in ['mnist', 'fmnist'] else 1)
            self.image_syn, self.label_syn = idm.synthesis(self.image_syn, self.label_syn,
                                                          self.consumer.train_set, self.local_data, self.start_time, test_loader = self.consumer.test_loader)
        elif general_args.synthesis_method.lower() == 'dc':
            dc = DC(self.num_classes, self.malicious, self.device, self.img_size, self.syn_hyperparams, channel=3 if general_args.dataset not in ['mnist', 'fmnist'] else 1)
            self.image_syn, self.label_syn = dc.synthesis(self.image_syn, self.label_syn,
                                                          self.consumer.train_set, self.local_data, self.start_time, test_loader = self.consumer.test_loader)
        elif general_args.synthesis_method.lower() == 'cafe':
            cafe = CAFE(self.num_classes, self.malicious, self.device, self.img_size, self.syn_hyperparams, channel=3 if general_args.dataset not in ['mnist', 'fmnist'] else 1)
            self.image_syn, self.label_syn = cafe.synthesis(self.image_syn, self.label_syn,
                                                          self.consumer.train_set, self.local_data, self.start_time, test_loader = self.consumer.test_loader)
        elif general_args.synthesis_method.lower() == 'dam':
            dam = DAM(self.num_classes, self.malicious, self.device, self.img_size, self.syn_hyperparams, channel=3 if general_args.dataset not in ['mnist', 'fmnist'] else 1)
            self.image_syn, self.label_syn = dam.synthesis(self.image_syn, self.label_syn,
                                                          self.consumer.train_set, self.local_data, self.start_time, test_loader = self.consumer.test_loader)
        else:
            raise ValueError('Unrecognized Synthesis Method !')


