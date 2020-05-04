import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd

from zoo.pytorch_yolo_v3.utils.datasets import pad_to_square, resize


class PDestreDataset(Dataset):
    def __init__(self, list_path=None, img_size=416, augment=True, multiscale=True, normalized_labels=False):
        self.image_path = r'f:\my\Prog\CV\Datasets\pdestre\images'
        self.annotation_path = r'f:\my\Prog\CV\Datasets\pdestre\annot_test'
        self.annotation_files = os.listdir(self.annotation_path)

        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def get_annotation_df(self, annotation_name):
        df = pd.read_csv(os.path.join(self.annotation_path, annotation_name), header=None).iloc[:, :7]
        df = df.rename(columns={0: 'frame_idx', 1: 'person_id', 2: 'left', 3: 'top', 4: 'width', 5: 'height', 6: 'sex'})
        return df

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        annotation_name = self.annotation_files[index % len(self.annotation_files)]
        annotation = self.get_annotation_df(annotation_name)


        # Choose random frame
        frame_idx = random.randint(1, annotation.frame_idx.max())
        annotation = annotation[annotation['frame_idx'] == frame_idx]
        image_file = os.path.join(self.image_path, annotation_name.split('.')[0], f'{frame_idx:03d}.jpg')


        # Extract image as PyTorch tensor
        try:
            img = transforms.ToTensor()(Image.open(image_file).convert('RGB'))
        except Exception as e:
            print(f'OS ERROR in {image_file}')
            raise e

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        # label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        # return annotation
        boxes_numpy = annotation[['sex', 'left', 'top', 'width', 'height']].values

        if self.annotation_path is not None:

            boxes = torch.from_numpy(boxes_numpy.reshape(-1, 5))

            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1])
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3])

            y1 = h_factor * (boxes[:, 2])
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4])

            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]

            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        # if self.augment:
        #     if np.random.random() < 0.5:
        #         img, targets = horisontal_flip(img, targets)

        return image_file, img, targets

    def collate_fn(self, batch):

        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.annotation_files)


if __name__ == '__main__':
    p = PDestreDataset()
    print(p[0])