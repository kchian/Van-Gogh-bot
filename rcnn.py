#!/usr/bin/env python
# coding: utf-8

# In[258]:


import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from backbone import *
import math
import img_lib
# from torchvision.references.detection.engine import train_one_epoch, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# In[259]:

def filter_results(res, topn=3):
    # boxes = []
    # labels = []
    # scores = []
    # i=0
    # for bbox,label,score in zip(res[0]["boxes"], res[0]["labels"], res[0]["scores"]):
    #     if thresh is not None and score > thresh:
    #         boxes.append(list(bbox.detach().cpu().numpy()))
    #         labels.append(label.item())
    #         scores.append(score.item())
    #     elif thresh is None and i < topn:
    #         boxes.append(list(bbox.detach().cpu().numpy()))
    #         labels.append(label.item())
    #         scores.append(score.item())
    #         i += 1
    # return boxes, labels, scores
    boxes = []
    labels = []
    scores = []

    keep = torchvision.ops.nms(res[0]["boxes"], res[0]["scores"], 0.3)
    for ind, (bbox,label,score) in enumerate(zip(res[0]["boxes"], res[0]["labels"], res[0]["scores"])):
        if ind in keep:
            boxes.append(list(bbox.detach().cpu().numpy()))
            labels.append(label.item())
            scores.append(score.item())
    return boxes, labels, scores


def get_rcnn(sd_path=None, backbone=None):
    sc_model = StrokeClassifier(img_dims=(224, 224))
    if backbone is not None:
        sc_model.load_state_dict(torch.load(backbone))

    sc_model.eval()
    backbone = sc_model.features
    backbone.out_channels = 32

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))


    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=9,
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone,
                    num_classes=len(img_lib.STROKES) + 1, # background?
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler,
                    # box_head=sc_model.head,
                    # box_fg_iou_thresh=0.9,
                    # box_bg_iou_thresh=0.3, 
                    image_mean=[0],
                    image_std=[1],
    )
    try:
        if sd_path is not None:
            model.load_state_dict(torch.load(sd_path))
    except RuntimeError:
        sd = torch.load(sd_path)
        print(sd)
        sd["rpn.head.conv.weight"] = sd.pop(["rpn.head.conv.0.0.weight"])
        sd["rpn.head.conv.bias"] = sd.pop(["rpn.head.conv.0.0.bias"])
        sc_model.load_state_dict(sd)
        
    return model

# https://github.com/pytorch/vision/issues/6192
def gauss_noise_tensor(img):
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    sigma = 0.002
    
    out = img + sigma * torch.randn_like(img)
    
    if out.dtype != dtype:
        out = out.to(dtype)
        
    return out


if __name__ == "__main__":
    model = get_rcnn()
    transform = transforms.Compose([
        transforms.ColorJitter(0.1,0.2,0.2),
        transforms.ElasticTransform(alpha=5.0),
        gauss_noise_tensor,
    ])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_ds = RCNNImgDataset("strokes/data.pkl", img_lib.STROKES, max_strokes=2, min_strokes=1, augmentation=transform)
    model = model.to(device)
    def collate_fn(batch):
        imgs = [i[0] for i in batch]
        labels = [i[1] for i in batch]
        return imgs, labels
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    lr_scheduler = None #torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    scaler = None


    # In[276]:


    for epoch in range(30):
        model.train()
        epoch_loss = 0
        for images, targets in tqdm(train_dl):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses_reduced = sum(loss for loss in loss_dict.values())
            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(losses_reduced)
                break

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                optimizer.step()
            epoch_loss += loss_value
            if lr_scheduler is not None:
                lr_scheduler.step()
        print(f"epoch: {epoch}, loss:{epoch_loss}")
    torch.save(model.state_dict(), "rcnn.pt")
