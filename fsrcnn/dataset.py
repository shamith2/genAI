# Dataset Loading and Transformations
# Dataset: DIV2K
# from https://data.vision.ee.ethz.ch/cvl/DIV2K/

# references from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

import os
import random

import torch
from torchvision.transforms import v2

import PIL


class DIV2K(torch.utils.data.Dataset):
    """
    DIV2K dataset
    compatible with torch.utils.data.DataLoader or can be used independently without DataLoader
    """

    def __init__(
            self,
            root_dir: str,
            upscale_factor: int,
            train_mode: str,
            in_channels: int = 3,
    ) -> None:
        self.in_channels = in_channels
        self.train_mode = train_mode

        if train_mode == 'train':
            self.root_dir = os.path.join(root_dir, 'DIV2K_train_HR')

        else:
            self.root_dir = os.path.join(root_dir, 'DIV2K_valid_HR')

        # smallest image size in training dataset
        self.hr_crop_size = (648, 648)
        self.lr_size = (self.hr_crop_size[0] // upscale_factor, self.hr_crop_size[1] // upscale_factor)

        # list of dataset image paths
        self.images = [os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir)]

    def __call__(
            self,
            batch_size: int = 1,
            shuffle: bool = False,
            drop_last: bool = False
    ):

        self.batch_size = batch_size

        if shuffle:
            random.shuffle(self.images)

        # index for iterating through dataset
        self.idx = -1
        self.end = self.__len__() // self.batch_size

        # if true, drop the last incomplete batch
        if drop_last:
            self.end -= 1

        return self

    def __len__(
            self
    ) -> int:
        """
        Returns: size of dataset
        """
        return len(self.images)

    def transform(
            self,
            image_mode: str
    ) -> v2.Compose:
        """
        Converting HR to LR images for training/validation
        Step 1: Crop High Resolution (HR) image to size n * (f ** 2); n is upscale_factor
        Step 1: Downsample High Resolution (HR) image to Low Resolution (LR) image by a factor n
        Step 2: Crop LR image into set of f x f sub-images with a stride k
        """

        if image_mode == 'in':
            return v2.Compose([
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True)
            ])

        # data augmentation to increase data for training and to improve model accuracy
        if self.train_mode in ['train', 'val']:
            if image_mode == 'hr':
                return v2.Compose([
                    v2.RandomCrop(size=self.hr_crop_size, padding=None), # scaling
                    v2.RandomVerticalFlip(p=0.5), # rotation
                    v2.ColorJitter( # color jitter
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.1
                    )
                ])

            elif image_mode == 'lr':
                return v2.Compose([
                    v2.Resize( # scaling
                        size=self.lr_size,
                        interpolation=v2.InterpolationMode.BICUBIC,
                        antialias=False
                    )
                ])

    def __getitem__(
            self,
            idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # transformations to apply to each image set
        input_transform = self.transform(image_mode='in')
        hr_transform = self.transform(image_mode='hr')
        lr_transform = self.transform(image_mode='lr')

        # convert image to tensor
        input_image = input_transform(PIL.Image.open(self.images[idx]))

        # add batch dim to image
        input_image = input_image.unsqueeze(0)

        # apply transformations on the images
        if self.train_mode in ['train', 'val']:
            hr_image = hr_transform(input_image)
            lr_image = lr_transform(hr_image)

        else:
            hr_image = None
            lr_image = lr_transform(input_image)

        # ignore hr_image when train_mode is 'test'
        return lr_image, hr_image

    def __getitems__(
            self,
            batch_idx: list[int]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:

        # transformations to apply to each image set
        input_transform = self.transform(image_mode='in')
        hr_transform = self.transform(image_mode='hr')
        lr_transform = self.transform(image_mode='lr')

        # convert images to tensors
        input_batch = [input_transform(PIL.Image.open(self.images[idx])) for idx in batch_idx]
        hr_batch, lr_batch = [], []

        # iterate through the set of training/validation images to preprocess
        for img in input_batch:
            # apply transformations on the images
            if self.train_mode in ['train', 'val']:
                hr_image = hr_transform(img)
                hr_batch.append(hr_image)
                lr_batch.append(lr_transform(hr_image))

            else:
                lr_batch.append(lr_transform(img))

        # ignore hr_batch when train_mode is 'test'
        return lr_batch, hr_batch

    def collate_fn(
            self,
            batch: list[list[torch.Tensor], list[torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Custom collate_tensor_fn for returning image and label of different sizes"""

        lr_images, hr_images = batch

        lr_batch = torch.stack(lr_images, dim=0)

        # hr_images might be empty
        if len(hr_images):
            hr_batch = torch.stack(hr_images, dim=0)

            return lr_batch, hr_batch

        else:
            return lr_batch, None

    def __iter__(
            self
    ):

        return self

    def __next__(
            self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Custom function to iterate over the dataset without torch.utils.data.DataLoader"""

        if self.idx < self.end:
            self.idx += 1

        else:
            self.idx = 0
            raise StopIteration

        # transformations to apply to each image set
        input_transform = self.transform(image_mode='in')
        hr_transform = self.transform(image_mode='hr')
        lr_transform = self.transform(image_mode='lr')

        # iterate through the set of training/validation images to preprocess
        i = self.idx * self.batch_size

        # convert images to tensors
        input_batch = [input_transform(PIL.Image.open(img_path)) for img_path in self.images[i: i + self.batch_size]]
        hr_batch = torch.empty((0, self.in_channels) + self.hr_crop_size)
        lr_batch = torch.empty((0, self.in_channels) + self.lr_size)

        for img in input_batch:
            # add batch dim to image
            img = img.unsqueeze(0)

            # apply transformations on the images
            if self.train_mode in ['train', 'val']:
                hr_image = hr_transform(img)
                hr_batch = torch.cat((hr_batch, hr_image), dim=0)
                lr_batch = torch.cat((lr_batch, lr_transform(hr_image)), dim=0)

            else:
                lr_batch = torch.cat((lr_batch, lr_transform(img)), dim=0)

        # ignore hr_batch when train_mode is 'test'
        return lr_batch, hr_batch


if __name__ == '__main__':
    root_dir = os.path.join(os.path.expanduser('~'), 'IPU', 'gen_ai', 'fsrcnn', 'datasets', 'div2k')

    dataset = DIV2K(root_dir=root_dir, upscale_factor=4, train_mode='val', in_channels=3)

    print('\nUsing dataset without torch.utils.data.DataLoader\n')

    for i, (images, labels) in enumerate(dataset(batch_size=32, shuffle=False, drop_last=False)):
        print(i, images.size(), labels.size())

        if i == 5:
            break
