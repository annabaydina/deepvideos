import colorsys

import numpy as np
import torch
import torch.utils
from torch.autograd import Variable
import torchvision.transforms as transforms
from zoo.pytorch_yolo_v3.models import Darknet, parse_data_config
from PIL import Image, ImageFont, ImageDraw

from zoo.pytorch_yolo_v3.utils.datasets import pad_to_square, resize
from zoo.pytorch_yolo_v3.utils.utils import non_max_suppression, load_classes

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



class YoloV3(object):
    model: torch.nn.Module

    def __init__(self, is_tiny=False, img_size=1024):
        if is_tiny:
            raise NotImplementedError("Tiny Yolov3 is not connected")

        model_def = '../zoo/pytorch_yolo_v3/config/yolov3.cfg'
        # weights_path = '../zoo/pytorch_yolo_v3/weights/yolov3.weights'
        weights_path = '../zoo/pytorch_yolo_v3/weights/darknet53.conv.74'
        coco_names = '../zoo/pytorch_yolo_v3/data/coco.names'

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Darknet(model_def).to(device)
        self.model.load_darknet_weights(weights_path)
        self.img_size = img_size
        self.class_names = load_classes(coco_names)

        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)

        self.pad = (0, 0, 0, 0)
        self.zoom = 0

    def tranform_image(self, image):
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        if isinstance(image, str):
            img = transforms.ToTensor()(Image.open(image).convert('RGB'))
        elif isinstance(image, Image.Image):
            img = transforms.ToTensor()(image.convert('RGB'))
        else:
            raise ValueError('hui')

        img, self.pad = pad_to_square(img, 0)
        self.zoom = img.shape[1] / self.img_size
        img = resize(img, self.img_size)
        _, padded_h, padded_w = img.shape
        img = Variable(img.unsqueeze(0).type(Tensor), requires_grad=False)
        return img

    def evaluate(self, image_file):
        self.model.eval()

        img = self.tranform_image(image_file)

        with torch.no_grad():
            outputs = self.model(img)
            print(outputs.shape)
            outputs = non_max_suppression(outputs, conf_thres=0.001, nms_thres=0.5)
            print(len(outputs), outputs[0].shape)

        data = plt.imread(image_file)
        print(data.shape)
        plt.imshow(data)
        ax = plt.gca()
        # plt.show()
        for sample_i in range(len(outputs)):
            if outputs[sample_i] is None:
                continue

            output = outputs[sample_i]
            pred_boxes = output[:, :4]
            pred_scores = output[:, 4]
            pred_labels = output[:, -1]
            print (output.shape)
            # return

            print(pred_scores.shape, pred_labels.shape)
            for i in range(len(pred_scores)):
                if pred_scores[i] < 0.8:
                    break
                x1, y1, x2, y2 = pred_boxes[i] * self.zoom
                x1 -= self.pad[0]
                x2 -= self.pad[0]
                y1 -= self.pad[2]
                y2 -= self.pad[2]

                width, height = x2 - x1, y2 - y1
                rect = Rectangle((x1, y1), width, height, fill=False, color='red')
                ax.add_patch(rect)
                label_name = self.class_names[int(pred_labels[i])]
                label = "%s (%.3f)" % (label_name, float(pred_scores[i]))
                plt.text(x1, y1, label, color='red',fontsize=6)
                print(f'Box:{pred_boxes[i]}, score={pred_scores[i]}, label={pred_labels[i]}')
        plt.show()

        return

    def evaluate_pil(self, image) -> Image:
        self.model.eval()

        img = self.tranform_image(image)

        with torch.no_grad():
            outputs = self.model(img)
            print(outputs.shape)
            outputs = non_max_suppression(outputs, conf_thres=0.001, nms_thres=0.5)
            print(len(outputs), outputs[0].shape)

        for sample_i in range(len(outputs)):
            if outputs[sample_i] is None:
                continue

            output = np.array(outputs[sample_i])
            pred_boxes = output[:, :4]
            pred_scores = output[:, 4]
            pred_labels = output[:, -1]

            font = ImageFont.truetype(font='UniversC.otf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = (image.size[0] + image.size[1]) // 300

            for i in range(len(pred_scores)):
                if pred_scores[i] < 0.5:
                    break
                x1, y1, x2, y2 = pred_boxes[i] * self.zoom
                x1 -= self.pad[0]
                x2 -= self.pad[0]
                y1 -= self.pad[2]
                y2 -= self.pad[2]

                # width, height = x2 - x1, y2 - y1

                class_index = int(pred_labels[i])
                label_name = self.class_names[class_index]
                label = "%s (%.3f)" % (label_name, float(pred_scores[i]))
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
                        outline=self.colors[class_index])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[class_index])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw

        return image
