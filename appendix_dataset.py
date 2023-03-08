import os
import random

import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset, DataLoader

up_list = ["446189", "676798", "716576", "417232"]
down_list = ["422737"]


def get_file_path(base_dir, split='train'):
    current_path = os.path.join(base_dir, split)
    image_paths, label_paths = [], []
    for sub_dir in os.listdir(os.path.join(current_path, 'images')):
        for file in os.listdir(os.path.join(current_path, 'images/{}'.format(sub_dir))):
            image_paths.append(os.path.join(current_path, 'images/{}/{}'.format(sub_dir, file)))
            label_paths.append(os.path.join(current_path, 'labels/{}/{}'.format(sub_dir, file)))
    return image_paths, label_paths


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image_result = []
    for item in image:
        item = np.rot90(item, k)
        image_result.append(item)
    image = np.stack(image_result, axis=0)
    label = np.rot90(label, k)
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-40, 40)
    result_images = []
    for item in image:
        item = ndimage.rotate(item, angle, order=0, reshape=False)
        result_images.append(item)
    image = np.stack(result_images, axis=0)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def random_gaussian(image, label):
    result_images = []
    for item in image:
        item = ndimage.gaussian_filter(item, sigma=1)
        result_images.append(item)
    image = np.stack(result_images, axis=0)
    return image, label


def random_shock(image, label):
    result_images = []
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    for item in image:
        item = cv2.filter2D(item, -1, kernel=kernel)
        result_images.append(item)
    image = np.stack(result_images, axis=0)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, random_convert=False):
        self.output_size = output_size
        self.random_convert = random_convert

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.random_convert:
            if random.random() > 0.5:
                image, label = random_rot_flip(image, label)
            if random.random() > 0.5:
                image, label = random_rotate(image, label)
            if random.random() > 0.5:
                image, label = random_gaussian(image, label)
            if random.random() > 0.5:
                image, label = random_shock(image, label)
        c, x, y = image.shape

        # scalar image to setting shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32))
        image = (image - torch.min(image)) / (torch.max(image) - torch.min(image) + 1e-20)  # normalize to 0-1
        label = torch.from_numpy(label.astype(np.int32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Appendix_Dataset_3slice(Dataset):
    def __init__(self, base_dir, split, transform=None, wl=-50, ww=50):
        self.transform = transform
        self.split = split
        self.data_dir = base_dir
        self.images, self.labels = get_file_path(base_dir, split)
        self.low, self.high = wl - ww / 2, wl + ww / 2  # calculate window level and window width

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = []
        image_path = self.images[idx]
        label_path = self.labels[idx]
        x, y = np.shape(np.load(image_path))
        image_file_name = os.path.basename(image_path)
        image_dir_name = os.path.dirname(image_path)
        sub_dir = image_dir_name.split('/')[-1]
        item = eval(image_file_name[:-4])
        label = np.load(label_path)
        for i in range(item - 1, item + 2):
            temp_path = os.path.join(image_dir_name, '{}.npy'.format(item))
            if os.path.exists(temp_path):
                image = np.load(temp_path)
                if sub_dir in up_list:
                    image = image[int(x * 0.2):int(x * 0.7), int(y * 0.1):int(y * 0.6)]
                elif sub_dir in down_list:
                    image = image[int(x * 0.5):int(x * 1.0), int(y * 0.1):int(y * 0.6)]
                else:
                    image = image[int(x * 0.4):int(x * 0.9), int(y * 0.1):int(y * 0.6)]
            else:
                image = np.zeros((x // 2, y // 2))  # fill zero slice
            image[image < self.low], image[image > self.high] = self.low, self.high
            image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-20)
            images.append(image)
        images = np.stack(images, axis=0)

        if sub_dir in up_list:
            label = label[int(x * 0.2):int(x * 0.7), int(y * 0.1):int(y * 0.6)]
        elif sub_dir in down_list:
            label = label[int(x * 0.5):int(x * 1.0), int(y * 0.1):int(y * 0.6)]
        else:
            label = label[int(x * 0.4):int(x * 0.9), int(y * 0.1):int(y * 0.6)]
        label[label > 0] = 1
        sample = self.transform({'image': images, 'label': label})
        sample['file_name'] = image_path
        return sample


if __name__ == '__main__':
    dataset = Appendix_Dataset_3slice('E:/appendix_test', split='test',
                                      transform=RandomGenerator((224, 224), False))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for idx, item in enumerate(dataloader):
        image, label, file_name = item['image'], item['label'], item['file_name']
        temp = image[0]
        cv2.imwrite('1.png', np.array(temp[1] * 255))
        break
