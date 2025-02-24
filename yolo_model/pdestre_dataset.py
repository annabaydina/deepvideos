import random
import os
import cv2
from torch.utils.data import Dataset
import pandas as pd
import colorsys
import numpy as np
import torch
import torch.utils
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm

from yolo_model.yolo_pdestre import YoloV3Pdestre
from zoo.pytorch_yolo_v3.utils.utils import non_max_suppression, xywh2xyxy, get_batch_statistics, ap_per_class
from zoo.pytorch_yolo_v3.utils.datasets import pad_to_square, resize, ListDataset


class PDestreDataset(Dataset):
    def __init__(self, image_path=None, annotation_path=None,
                 img_size=416, augment=True, multiscale=True,
                 normalized_labels=False):
        if image_path is None:
            self.image_path = r'f:\my\Prog\CV\Datasets\pdestre\images'
        else:
            self.image_path = image_path
        if annotation_path is None:
            self.annotation_path = r'f:\my\Prog\CV\Datasets\pdestre\annotation'
        else:
            self.annotation_path = annotation_path

        self.image_format = 'jpg'  # 'mp4'
        self.annotation_files = os.listdir(self.annotation_path)
        self.img_size = img_size
        self.max_objects = 1000
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.bootstrap_factor = 100
        self.cropping = True
        self.crop_size = 832
        self.random_seed = 16
        random.seed(self.random_seed)

        self.current_video_idx = -1
        self.capturer = None

    def get_annotation_df(self, annotation_name):
        df = pd.read_csv(os.path.join(self.annotation_path, annotation_name), header=None).iloc[:, :7]
        df = df.rename(columns={0: 'frame_idx', 1: 'person_id', 2: 'left', 3: 'top', 4: 'width', 5: 'height', 6: 'sex'})
        return df

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        video_idx = index // self.bootstrap_factor
        # frame_idx = index % self.bootstrap_factor

        annotation_name = self.annotation_files[video_idx % len(self.annotation_files)]
        annotation = self.get_annotation_df(annotation_name)

        # Choose random frame
        frame_idx = random.randint(1, annotation.frame_idx.max())

        annotation = annotation[annotation['frame_idx'] == frame_idx]

        if self.image_format == 'mp4':
            if video_idx != self.current_video_idx:
                video_path = os.path.join(self.image_path, annotation_name.split('.')[0] + '.MP4')
                self.capturer = cv2.VideoCapture(video_path)
            self.capturer.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
            res, frame = self.capturer.read()
            if not res:
                raise ValueError(f"Can't get frame from {video_path} #{frame_idx}")

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        elif self.image_format == 'jpg':
            image_file = os.path.join(self.image_path, annotation_name.split('.')[0], f'{frame_idx:04d}.jpg')
            img = Image.open(image_file)

        if img.size[0] < self.crop_size or img.size[1] < self.crop_size:
            raise ValueError(f'Image {image_file} size {img.size} is less than crop size: {self.crop_size}')
        xc = random.randint(0, img.size[0] - 1 - self.crop_size)
        yc = random.randint(0, img.size[1] - 1 - self.crop_size)

        crop = (xc, yc, xc + self.crop_size, yc + self.crop_size)
        img = img.crop(crop)
        img = transforms.ToTensor()(img.convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        pad = [float(x) for x in pad]
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        # label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        # return annotation
        boxes_numpy = annotation[['sex', 'left', 'top', 'width', 'height']].values

        if self.annotation_path is not None:
            boxes = torch.from_numpy(boxes_numpy.astype(np.float32).reshape(-1, 5))

            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - xc)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] - xc)

            y1 = h_factor * (boxes[:, 2] - yc)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] - yc)

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

            ids = torch.all(((boxes[:, 1:3] > 0) & (boxes[:, 1:3] < 1)), axis=1)
            boxes = boxes[ids]
            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

            # Apply augmentations
            # if self.augment:
            #     if np.random.random() < 0.5:
            #         img, targets = horisontal_flip(img, targets)

        if self.image_format == 'mp4':
            original_file = (video_path, frame_idx)
        elif self.image_format == 'jpg':
            original_file = image_file

        return original_file, img, targets

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
        return len(self.annotation_files) * self.bootstrap_factor


def visualize_dataset(image_tensor, targets):
    # _, image_tensor, targets = dataset[idx]

    image = transforms.ToPILImage()(image_tensor)

    hsv_tuples = [(x / 80, 1., 1.)
                  for x in range(80)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))

    output = np.array(targets)
    pred_boxes = output[:, 2:]
    pred_labels = output[:, 1]

    font = ImageFont.truetype(font='UniversC.otf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    for i in range(len(pred_boxes)):

        cx, cy, w, h = pred_boxes[i]

        x1 = (cx - 0.5 * w) * image.size[0]
        x2 = (cx + 0.5 * w) * image.size[0]
        y1 = (cy - 0.5 * h) * image.size[1]
        y2 = (cy + 0.5 * h) * image.size[1]

        # width, height = x2 - x1, y2 - y1

        class_index = int(pred_labels[i])
        label = str(int(class_index))
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = y1, x1, y2, x2
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for j in range(thickness):
            draw.rectangle(
                [left + j, top + j, right - j, bottom - j],
                outline=colors[class_index])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[class_index])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
    image.show()
    return image


def evaluate_pdestre(yoloPdestre: YoloV3Pdestre, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model = yoloPdestre.model
    model.eval()

    # Get dataloader
    validation_video_path = r'f:\my\Prog\CV\Datasets\pdestre\images_eval'
    validation_annotation_path = r'f:\my\Prog\CV\Datasets\pdestre\annotation_eval'
    dataset = PDestreDataset(validation_video_path, validation_annotation_path, img_size=img_size, augment=False,
                             multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm(dataloader, desc="Detecting objects")):
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = imgs.type(Tensor)
        imgs.requires_grad = False

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == '__main__':
    # d = ListDataset(r'f:\my\Prog\CV\Datasets\coco\trainvalno5k.txt')
    # _, i, t = d[2234]
    # visualize_dataset(i, t)
    # exit()

    # print(d[0])
    p = PDestreDataset()
    #
    # _, img, targets = p[1225]
    # visualize_dataset(img, targets)
    #
    # exit()

    dataloader = torch.utils.data.DataLoader(
        p,
        batch_size=8,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=p.collate_fn,
    )
    for (_, imgs, targets) in dataloader:
        print(imgs.shape, targets.shape)
        for i, image_tensor in enumerate(imgs):
            visualize_dataset(image_tensor, targets[targets[:, 0] == i])
        break
    # print(p[0])
