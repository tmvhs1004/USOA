import torch.nn as nn
import torch
import Config as cfg

import math

def c_IOU(box1, box2, eps=1e-7) :
    # Get the coordinates of bounding boxes
    # b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    # b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2

    # print(b1_x1[0])
    # print(b1_y1[0])
    # print(b1_x2[0])
    # print(b1_y2[0])
    #
    # print(box2[0,...])
    #
    # raise NotImplementedError
    b1_x1, b1_x2 = box1[0].reshape(1,1), box1[2].reshape(1,1)
    b1_y1, b1_y2 = box1[1].reshape(1,1), box1[3].reshape(1,1)


    b2_x1, b2_x2 = box2[:, 0], box2[:, 2]
    b2_y1, b2_y2 = box2[:, 1], box2[:, 3]

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))

    return iou - (rho2 / c2 + v * alpha)



