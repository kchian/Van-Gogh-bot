
import cv2
from scipy import ndimage
import numpy as np
from scipy import ndimage
import random
import torch
import pickle
import matplotlib.pyplot as plt
import os
# top_left = [ 0.32886487, -0.25668124,  0.10390387]
# bottom_right = [ 0.59114306,  0.28672476,  0.10615327]
# STROKES = ["", "straight", "convex", "concave", "squiggle"]
STROKES = ["", "spiral", "box", "triangle", "plus", "circle", "star"]
COLORS = np.random.uniform(0, 255, size=(len(STROKES), 3))
H = 290
W = 615


def get_random_stroke(stroke, strokes_dir="strokes"):
    imgs = [os.path.join(strokes_dir, stroke, i) for i in os.listdir(os.path.join(strokes_dir, stroke))]
    return cv2.imread(random.choice(imgs), cv2.IMREAD_UNCHANGED)

def draw_boxes(boxes, labels, scores, image):
    # read the image with OpenCV
    if type(image) == torch.Tensor:
        image = (image.numpy() * 255).astype(np.uint8)
    if image.shape[0] == 1: # channels
        image = image.transpose(1,2,0)
    if image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, f"{STROKES[labels[i]]}:{scores[i]:.2f}", (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType=cv2.LINE_AA)
    
    return image

def prep_tensor(img):
    # prep image for inference
    # img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)[:,:,None]
    img = img.transpose(2,0,1)
    return torch.Tensor(img/255).unsqueeze(0)

def prep_img(img, base=None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = ndimage.rotate(img, 2, order=0)[260:550,625:1240,:]
    if base is not None:
        img = cv2.absdiff(base, img)
        img[img < 10] = 0
        img[img > 50] = 255
    return img

def remove_bg(img, base=None):
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    if base is not None:
        img = cv2.absdiff(base, img)
        img[img < 10] = 0
        img[img > 50] = 255
    else:
        img[img < 130] = 0
        img[img >= 160] = 255
    return img

def halve_img(img, ret_ind):
    ind = img.shape[1] // 2
    if ret_ind == 0:
        return img[:, :ind]
    elif ret_ind == 1:
        return img[:, ind:]
    else:
        raise ValueError

def do_transform_vanilla(img, base):
    # Set the size of the frame

    label = np.array([[0, 0], [img.shape[0], img.shape[1]]])
    scale = random.uniform(min(base.shape[:2]) / max(img.shape) * 0.2, min(base.shape[:2]) / max(img.shape) * 0.6)

    if img.shape[1] > img.shape[0]:
        diff = img.shape[1] - img.shape[0]
        img = np.pad(img, ((diff//2, diff//2), (0, 0), (0, 0)))
        label[:,0] += diff//2
    if img.shape[0] > img.shape[1]:
        diff = img.shape[0] - img.shape[1]
        img = np.pad(img, ((0, 0), (diff//2, diff//2), (0, 0)))
        label[:,1] += diff//2
    
    target_size = (np.array(img.shape) * scale).astype(int)
    diff = target_size - img.shape

    if scale > 1:
        img = np.pad(img, ((diff[0]//2, diff[0]//2), (diff[1]//2, diff[1]//2), (0, 0)))
        label[:,0] += diff[0]//2
        label[:,1] += diff[1]//2
    if scale < 1:
        label = label * scale 
        label += (np.array(img.shape[:2]) * (1-scale)//2)
        label = label.astype(int)
#         label = np.array([[0, 0], target_size[:2]])


    center = (img.shape[0]//2, img.shape[1]//2)
    angle = random.randint(-30, 30)
    rot_mat = cv2.getRotationMatrix2D( center, angle, scale )
    out = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))

    transform_label = label

    # overlay on base img
    base = cv2.cvtColor(base, cv2.COLOR_RGB2RGBA)
    offset_h = random.randint(0, base.shape[0] - out.shape[0])
    offset_w = random.randint(0, base.shape[1] - out.shape[1])  
    base[offset_h:offset_h+out.shape[0], offset_w:offset_w+out.shape[1]] = np.where(out[:,:,3,None] == 0, base[offset_h:offset_h+out.shape[0], offset_w:offset_w+out.shape[1]], out)

    transform_label[:,0] += offset_h
    transform_label[:,1] += offset_w
    transform_label = transform_label.astype(int)
#     base[transform_label[0][0]:transform_label[0][0] + 4, transform_label[0][1]:transform_label[0][1]+4] = np.array([255,0,255,255])
#     base[transform_label[1][0]:transform_label[1][0] + 4, transform_label[1][1]:transform_label[1][1]+4] = np.array([255,0,255,255])
    return base, transform_label


def get_realistic_image(strokes="strokes/data.pkl"):
    with open(strokes, "rb") as f:
        data = pickle.load(f)
    base = data["base"]
    for i in range(3):
        item = data[STROKES[random.randint(1, len(STROKES)-1)]]
        base, coord_labels = do_transform_vanilla(item["img"], base)
    return base
