import torch
import torch.utils
from torch.autograd import Variable
import torchvision.transforms as transforms
from zoo.pytorch_yolo_v3.models import Darknet, parse_data_config
from PIL import Image

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
        weights_path = '../zoo/pytorch_yolo_v3/weights/yolov3.weights'
        coco_names = '../zoo/pytorch_yolo_v3/data/coco.names'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Darknet(model_def).to(device)
        self.model.load_darknet_weights(weights_path)
        self.img_size = img_size
        self.class_names = load_classes(coco_names)
        self.pad = (0, 0, 0, 0)
        self.zoom = 0

    def tranform_image(self, image_file):
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        img = transforms.ToTensor()(Image.open(image_file).convert('RGB'))
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
