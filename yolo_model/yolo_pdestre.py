import colorsys

from yolo_model.yolov3 import YoloV3
import numpy as np
import torch
import torch.utils
from torch.autograd import Variable
import torchvision.transforms as transforms
from zoo.pytorch_yolo_v3.models import Darknet, parse_data_config
from PIL import Image, ImageFont, ImageDraw

from zoo.pytorch_yolo_v3.utils.datasets import pad_to_square, resize
from zoo.pytorch_yolo_v3.utils.utils import non_max_suppression, load_classes


class YoloV3Pdestre(YoloV3):
    def __init__(self, is_tiny=False, img_size=1024):

        model_def = 'f:/my/Prog/CV/deepvideos/yolo_model/yolo_pdestre_config/yolov3_pdestre.cfg'

        weights_path = '../zoo/pytorch_yolo_v3/weights/darknet53.conv.74'
        class_names = ['male', 'female', 'unknown']


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Darknet(model_def).to(device)
        self.model.load_darknet_weights(weights_path)
        self.img_size = img_size
        self.class_names = class_names

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