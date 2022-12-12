import pickle
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import matplotlib.pyplot as plt
import img_lib

class StrokeClassifier(nn.Module):

    def __init__(self, img_dims=(img_lib.H, img_lib.W)):
        super(StrokeClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, 3),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(4, 8, 3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(3),
            nn.Conv2d(8, 16, 3),
            # nn.ReLU(),
            # nn.BatchNorm2d(16),
            nn.MaxPool2d(3),
            nn.Conv2d(16, 32, 3),
        )
        dummy = torch.zeros(img_dims).unsqueeze(0).unsqueeze(0)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(self.features(dummy).shape), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, len(img_lib.STROKES)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x

class ImgDataset(Dataset): # Basically deprecated, doesn't work with rcnn
    def __init__(self, data_path, strokes, size=(224, 224), augmentation=None, return_coords=False, length=1000):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
        self.return_coords = return_coords
        self.augmentation = augmentation
        self.size = size
        self.strokes = strokes
        self.length = length
        self.square_base = cv2.resize(self.data["base"], size)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        crop_h = random.randint(0, self.data["base"].shape[0] - self.size[0])
        crop_w = random.randint(0, self.data["base"].shape[1] - self.size[1])
        base = self.data["base"][crop_h:crop_h + self.size[0], crop_w:crop_w + self.size[1]]
        index = random.randint(1, len(self.strokes) - 1)
        item = self.data[self.strokes[index]]
        stroke = img_lib.get_random_stroke(self.strokes[index])
        img, coord_labels = img_lib.do_transform_vanilla(stroke, base)
        img = img_lib.remove_bg(img, self.square_base)
        img = img[:,:,:3].transpose(2,0,1)
        img = self.augmentation(torch.Tensor(img)/255) * 255
        img = cv2.cvtColor(img.numpy().transpose(1,2,0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return torch.Tensor(img/255).unsqueeze(2).permute(2,0,1), item["label"]
    


class RCNNImgDataset(Dataset):
    def __init__(self, data_path, strokes, size=(224, 224), augmentation=None, return_coords=False, length=1000, min_strokes=2, max_strokes=5):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
        self.return_coords = return_coords
        self.augmentation = augmentation
        self.size = size
        self.strokes = strokes
        self.length = length
        self.min_strokes = min_strokes
        self.max_strokes = max_strokes
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # random.seed(idx)
        bbox_labels = []
        n_strokes = random.randint(self.min_strokes, self.max_strokes)
        labels = []
        img = self.data["base"]
        for i in range(n_strokes):
            index = random.randint(1, len(self.strokes) - 1) # index 0 is background
            item = self.data[self.strokes[index]]
            stroke = img_lib.get_random_stroke(self.strokes[index])
            img, coord_labels = img_lib.do_transform_vanilla(stroke, img)
            bbox_labels.append([
                coord_labels[0][1],
                coord_labels[0][0],
                coord_labels[1][1],
                coord_labels[1][0]
            ])
            labels.append(index)

        img = img[:,:,:3].transpose(2,0,1)
        if self.augmentation is not None:
            img = self.augmentation(torch.Tensor(img)/255) * 255
            img = img.numpy()

        img = img_lib.remove_bg(img.transpose(1,2,0).astype(np.uint8), self.data["base"])

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        out_label = {
            "boxes": torch.Tensor(bbox_labels),
            "labels": torch.Tensor(labels).type(torch.int64),
        }
        return torch.Tensor(img/255).unsqueeze(2).permute(2,0,1), out_label

    

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ColorJitter(0.2,0.3,0.3),
        # transforms.RandomRotation(degrees=(0, 180)),
        transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.2), scale=(0.8, 1)),
        transforms.ElasticTransform(alpha=10.0),
    ])
    train_ds = ImgDataset("strokes/data.pkl", strokes=img_lib.STROKES, augmentation=transform)
    
    model = StrokeClassifier(img_dims=(224, 224))
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(30):
        correct = 0
        tot = 0
        for batch in tqdm(train_dl):
            imgs = batch[0]
            labels = batch[1]
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            correct += (torch.argmax(output, dim=1) == labels).sum()
            tot += len(labels)
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch}, acc {correct/tot}")
    torch.save(model.state_dict(), "backbone.pt")