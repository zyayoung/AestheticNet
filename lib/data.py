from torchvision.datasets import VisionDataset
import os
import torch
from tqdm import tqdm
from PIL import Image
import cv2


class AVA(VisionDataset):

    def __init__(self, root='ava', transform=None, target_transform=None):
        super(AVA, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.root = root
        self.training_file = os.path.join(self.root, 'AVA.txt')
        if not self._check_exists():
            raise RuntimeError('Dataset not found.')
        
        with open(self.training_file) as f:
            df = [[int(i) for i in l.split()] for l in f.readlines()]
        
        avail_files = set(os.listdir(os.path.join(self.root, 'images')))
        self.targets = []
        self.data = []
        for d in df:
            if str(d[1])+'.jpg' in avail_files:
                self.data.append(d[1])
                self.targets.append([int(i) for i in d[2:12]])
        
        if not os.path.exists(os.path.join(self.root, 'processed')):
            os.mkdir(os.path.join(self.root, 'processed'))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        ImageID, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if os.path.exists(self.processed_im(ImageID)):
            img = cv2.imread(self.processed_im(ImageID))
        else:
            img = cv2.imread(self.raw_im(ImageID))
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(self.processed_im(ImageID), img)

        img = torch.tensor(img[..., [2, 1, 0]]).permute(2, 0, 1).float() / 255.
        if self.transform is not None:
            img = self.transform(img)
        target = torch.tensor(target).float()

        return img, target

    def __len__(self):
        return len(self.data)

    def raw_im(self, im):
        return os.path.join(self.root, 'images', str(im)+".jpg")

    def processed_im(self, im):
        return os.path.join(self.root, 'processed', str(im)+".jpg")

    def _check_exists(self):
        return os.path.exists(self.training_file)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
