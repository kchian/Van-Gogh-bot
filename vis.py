import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import learn
import math
import torch

def plot_path(traj):

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter3D(traj[:, 0], traj[:, 1], traj[:, 2], cmap='Greens');
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(traj[:, 0], label="x")
    ax.plot(traj[:, 1], label="y")
    ax.plot(traj[:, 2], label="z")
    ax.legend()

def plot_f(f, fitted=None):
    fig, ax = plt.subplots(1,1,  figsize=(12, 8))
    ax.plot(f, label="x")
    if fitted is not None:
        ax.plot(fitted, label="fitted")
    ax.legend()

def plot_drawing(trajectory, fitted=None):
    fig, ax = plt.subplots(1,1,  figsize=(12, 8))
    ax.scatter(trajectory[:, 0],trajectory[:, 1], label="original")
    if fitted is not None:
        ax.scatter(fitted[:, 0],fitted[:, 1], label="fitted")
    # plt.xlim(0.32886487, 0.59114306)
    # plt.ylim(-0.25668124, 0.28672476)
    ax.set_aspect('equal')
    fig.tight_layout()
    ax.legend()
    return ax

# https://www.askpython.com/python/normal-distribution
def normal_dist(x, mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

def plot_distributions(mu, h):
    fig, ax = plt.subplots(1,1,  figsize=(12, 8))
    sample = 10000
    for i, j in zip(mu, h):
        x = [t / sample for t in range(sample)]
        y = [normal_dist(x_i, i, j) for x_i in x]
        ax.plot(y)

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
