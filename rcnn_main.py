import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import math
# from torchvision.references.detection.engine import train_one_epoch, evaluate
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import rcnn
import img_lib
import backbone

import importlib
importlib.reload(rcnn)
importlib.reload(img_lib)
# importlib.reload(backbone)
import argparse
import vis

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="The input file to process")
    # parser.add_argument("-n", type=int, help="Number of shapes in img")
    args = parser.parse_args()

    model = rcnn.get_rcnn("rcnn.pt")
    model.eval()
    img = cv2.imread(args.input_file)

    with open("strokes/data.pkl","rb") as f:
        data = pickle.load(f)
        base = data['base']
        img = img_lib.prep_img(img)
        img = img_lib.remove_bg(img, base)
        # img = img_lib.halve_img(img, 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)[:,:,None]
        out = model(img_lib.prep_tensor(img))
        boxes, preds, scores = rcnn.filter_results(out) 
        # img = img_lib.draw_boxes(boxes, preds, scores, img)
        # plt.imshow(img)
        # plt.show()

    with open("model_out.txt", "w") as f:
        for box, pred, score in zip(boxes, preds, scores):
            print(f"pred: {pred}")
            f.write(img_lib.STROKES[pred] + " " + str(" ".join([str(int(i)) for i in box])))
            f.write("\n")