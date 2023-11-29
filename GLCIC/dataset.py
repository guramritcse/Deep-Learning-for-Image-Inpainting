import torch.utils.data as data
from os import listdir
from ..utils.tools import image_loader, is_image_file, normalize
import os
import torchvision.transforms as transforms

"""
Dataset that loads training and test data.
"""
class Dataset(data.Dataset):
    def __init__(self, data_path, image_shape, random_crop=True, return_name=False):
        super(Dataset, self).__init__()
        self.samples = [x for x in listdir(data_path) if is_image_file(x)]
        self.data_path = data_path
        self.image_shape = image_shape[:-1]
        self.random_crop = random_crop
        self.return_name = return_name

    def __getitem__(self, index):
        path = os.path.join(self.data_path, self.samples[index])
        img = image_loader(path)

        if self.random_crop:
            img_width, img_height = img.size
            if img_height < self.image_shape[0] or img_width < self.image_shape[1]:
                img = transforms.Resize(min(self.image_shape))(img)
            img = transforms.RandomCrop(self.image_shape)(img)
        else:
            img = transforms.Resize(self.image_shape)(img)
            img = transforms.RandomCrop(self.image_shape)(img)

        img = transforms.ToTensor()(img) 
        img = normalize(img)

        if self.return_name:
            return self.samples[index], img
        else:
            return img

    def __len__(self):
        return len(self.samples)
