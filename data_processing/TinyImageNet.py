import os
import zipfile
import requests
from tqdm import tqdm
from PIL import Image
import torch
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
import numpy as np
from hyperparams.log import logger
from hyperparams.general_params import general_args

class TinyImageNet(VisionDataset):
    """Custom Tiny-ImageNet Dataset with Random Class Selection and Mapping."""

    base_folder = 'tiny-imagenet-200'
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False,
                 num_classes=None, random_seed=2, save_mapping=False, start_time = None):
        """
        Args:
            root (string): Root directory of dataset where ``tiny-imagenet-200`` folder exists or will be saved to if download is set to True.
            split (string): One of 'train', 'val'.
            transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory.
            num_classes (int, optional): Number of classes to randomly select. If None, use all classes.
            random_seed (int, optional): Seed for random class selection.
            save_mapping (bool, optional): If True, save the inverse mapping to a .pt file.
        """
        super(TinyImageNet, self).__init__(root, transform=transform, target_transform=target_transform)
        
        self.split = split
        self.data = []  # List to hold image data as numpy arrays
        self.targets = []  # List to hold target labels
        self.class_to_idx = {}
        self.classes = []
        self.mapping = None  # new_idx -> original_idx
        self.inverse_mapping = None  # original_idx -> new_idx
        self.start_time = start_time

        if download:
            self.download()
        
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it.")
        
        # Load all classes
        train_dir = os.path.join(self.root, self.base_folder, 'train')
        all_classes = sorted(os.listdir(train_dir))
        total_classes = len(all_classes)
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Select classes
        if num_classes is not None and num_classes < total_classes:
            selected_class_indices = np.random.permutation(total_classes)[:num_classes]
            selected_class_indices.sort()
            selected_classes = [all_classes[idx] for idx in selected_class_indices]
            self.classes = selected_classes
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
            
            # Create mappings
            self.mapping = {new_idx: original_idx for new_idx, original_idx in enumerate(selected_class_indices)}
            # self.inverse_mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(selected_class_indices)}
            
            logger.info_once(f"Selected {num_classes} classes: {self.classes}")
            logger.info_once("Class mapping (new_idx -> original_idx):")
            for new_idx, original_idx in self.mapping.items():
                logger.info_once(f"{new_idx} -> {original_idx} ({self.classes[new_idx]})")
            if save_mapping:
                mapping_path = os.path.join(f"../results/{general_args.synthesis_method}/{general_args.dataset}/",f"{start_time[1]:02}{start_time[2]:02}{start_time[3]:02}{start_time[4]:02}{start_time[5]:02}", 'mapping.pt')
                torch.save(self.mapping, mapping_path)
                logger.info_once(f"Inverse mapping saved to {mapping_path}")
        else:
            self.classes = all_classes
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
            logger.info(f"Using all {total_classes} classes.")


        # Load data based on split
        if split == 'train':
            self._load_train()
        elif split == 'val':
            self._load_val()
        else:
            raise ValueError("split must be 'train' or 'val'")
        
        # Convert data and targets to np.array
        self.data = np.array(self.data)  # Convert list of image arrays to numpy array (N, H, W, C)
        self.targets = np.array(self.targets)  # Convert targets list to numpy array

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.base_folder))
    
    def download(self):
        """Download and extract the Tiny-ImageNet dataset."""
        if self._check_exists():
            logger.info('Dataset already downloaded and extracted.')
            return

        os.makedirs(self.root, exist_ok=True)
        zip_path = os.path.join(self.root, self.filename)

        # Download the dataset with a progress bar
        logger.info("Downloading Tiny-ImageNet dataset...")
        response = requests.get(self.url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        with open(zip_path, 'wb') as f, tqdm(
            desc=zip_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
        ) as bar:
            for data in response.iter_content(block_size):
                f.write(data)
                bar.update(len(data))
        
        # Extract the dataset
        logger.info("Extracting Tiny-ImageNet dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root)
        
        os.remove(zip_path)
        logger.info("Download and extraction complete.")

    def _load_train(self):
        """Load the training split, optionally filtering by selected classes."""
        train_dir = os.path.join(self.root, self.base_folder, 'train')
        for cls in self.classes:
            cls_folder = os.path.join(train_dir, cls, 'images')
            if not os.path.isdir(cls_folder):
                logger.info(f"Warning: Class folder {cls_folder} does not exist.")
                continue
            for img_name in os.listdir(cls_folder):
                img_path = os.path.join(cls_folder, img_name)
                try:
                    with Image.open(img_path).convert('RGB') as img:
                        img = img.resize((64, 64))  # Resize to a consistent size if necessary
                        img_np = np.array(img)
                        self.data.append(img_np)
                        self.targets.append(self.class_to_idx[cls])
                except Exception as e:
                    logger.info(f"Error loading image {img_path}: {e}")

    def _load_val(self):
        """Load the validation split, assuming validation images are organized into class-specific folders."""
        val_dir = os.path.join(self.root, self.base_folder, 'val')
        possible_class_dirs = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]

        for cls in self.classes:
            cls_folder = os.path.join(val_dir, cls)
            if not os.path.isdir(cls_folder):
                logger.info(f"Warning: Class folder {cls_folder} does not exist.")
                continue
            for img_name in os.listdir(cls_folder):
                img_path = os.path.join(cls_folder, img_name)
                try:
                    with Image.open(img_path).convert('RGB') as img:
                        img = img.resize((64, 64))  # Resize to a consistent size if necessary
                        img_np = np.array(img)
                        self.data.append(img_np)
                        self.targets.append(self.class_to_idx[cls])
                except Exception as e:
                    logger.info(f"Error loading image {img_path}: {e}, we will try to load the img with another strategy")
        
        if len(self.data) == 0:
            self._load_val_alternative()

    def _load_val_alternative(self):
        """Load the validation split, assuming validation images are organized into class-specific folders."""
        val_dir = os.path.join(self.root, self.base_folder, 'val')
        val_annotations_dir = os.path.join(val_dir, 'val_annotations.txt')
        with open(val_annotations_dir, 'r') as f:
            for line in f:
                img_name, cls = line.strip().split('\t')[:2]
                if cls not in self.class_to_idx:
                    continue
                img_path = os.path.join(val_dir, 'images', img_name)
            
                with Image.open(img_path).convert('RGB') as img:
                    img = img.resize((64, 64))  # Resize to a consistent size if necessary
                    img_np = np.array(img)
                    self.data.append(img_np)
                    self.targets.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """Retrieve an image and its target."""
        img, target = self.data[index], self.targets[index]
        
        # 转换为 PIL 图像以应用变换
        img = Image.fromarray(img)
        
        # 应用变换
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def get_mapping(self):
        """Return the mapping from new class indices to original class indices."""
        return self.mapping

    def get_inverse_mapping(self):
        """Return the inverse mapping from original class indices to new class indices."""
        return self.inverse_mapping

    def extra_repr(self):
        return f"Split: {self.split}, Number of classes: {len(self.classes)}"

