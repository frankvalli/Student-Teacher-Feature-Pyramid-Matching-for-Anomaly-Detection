import os
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as T


def get_transforms(size, mean, std) -> transforms.Compose:
    '''Get transforms
    
    Args:
        size (int): size to resize images to
        mean (list[float]): means to normalize images
        std (list[float]): stds to normalize images

    Returns:
        transform (transforms.Compose): transform.Compose instance
    '''
    transform = transforms.Compose(
        [transforms.Resize((size, size)),
         transforms.ToTensor(),
         transforms.Normalize(mean=mean,std=std)])
    return transform


class AnomalyDataset(Dataset):
    '''Dataset class
    
    Args:
        root_dir (str): data directory
        transform (transforms.Compose): transforms
    '''

    def __init__(self, root_dir: str, transform: transforms.Compose) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.phase = os.path.basename(root_dir)
        self.images = sorted(glob.glob(self.root_dir+'/*/*.png'))

    def __len__(self) -> int:
        '''Get length of dataset
        
        Returns:
            int: length of dataset
        '''
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        '''Get dataset item at index `index`

        Args:
            idx (int): item index

        Returns:
            dict[str, torch.Tensor]: dictionary of tensor image, mask and label
        '''
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)

        class_path, filename = os.path.split(img_path)
        class_name = os.path.basename(class_path)

        if class_name == 'good':
            mask = torch.zeros(img_tensor.shape[1:], dtype=torch.int)
            label = torch.tensor(0.)
        else:
            mask_folder = class_path.replace(self.phase, 'ground_truth')
            mask_filename = filename.split('.')[0]+'_mask.png'
            mask_path = os.path.join(mask_folder, mask_filename)
            mask = Image.open(mask_path)
            mask = T.resize(mask, img_tensor.shape[1:])
            mask = T.to_tensor(mask).type(torch.int).squeeze()
            label = torch.tensor(1.)

        sample = {'image': img_tensor,
                  'mask': mask,
                  'label': label}
        return sample