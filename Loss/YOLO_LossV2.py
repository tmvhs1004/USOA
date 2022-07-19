import torch
import torch.nn as nn
import Config as cfg
import Util.BBox as BX
import math

class YoloLossV2(nn.Module) :
    def __init__(self):
        super(YoloLossV2, self).__init__()

        # Box = MSE , Obj =
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        # self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.seg_loss = nn.CrossEntropyLoss()


        self.weight_obj = 1
        self.weight_noobj = 10
        self.weight_box=3
        self.weight_seg = 2

    def forward(self, prd_f, target_f, indices_f, f_idx, seg, seg_label):

        prd_f[...,0:2] = prd_f[...,0:2].sigmoid()

        # CIOU

        b, a, h, w = self.split_indices(indices_f)
        ps = prd_f[b,a,h,w]
        ts = target_f[b,a,h,w, 1:5]

        px = cfg.LEN_WIDTH[f_idx] * (ps[:, 0] + torch.tensor(w).to(cfg.DEVICE))
        py = cfg.LEN_HEIGHT[f_idx] * (ps[:, 1] + torch.tensor(h).to(cfg.DEVICE))

        anchor = torch.tensor(cfg.ANCHORS).to(cfg.DEVICE)
        pcw = anchor[f_idx, a,0] * torch.exp(ps[:, 2])  # YOLOV3
        pch = anchor[f_idx, a,1] * torch.exp(ps[:, 3])  # YOLOv3

        px= px.reshape(-1,1)
        py = py.reshape(-1, 1)
        pcw = pcw.reshape(-1, 1)
        pch = pch.reshape(-1, 1)


        pbox = torch.cat([px, py, pcw, pch], dim=1)

        # x1, x2 =px - pcw /2, px + pcw /2
        # y1, y2 = py - pch / 2, py + pch / 2
        #
        #
        #
        #
        # # print(px[:5])
        # # print(py[:5])
        # # print(pcw[:5])
        # # print(pch[:5])
        # #print(pbox[:2])
        # print(torch.cat([x1[0], y1[0], x2[0], y2[0]]))
        # print(ts[0])
        #
        # raise NotImplementedError

        ciou = BX.bbox_iou(pbox, ts.to(cfg.DEVICE))
        lbox = (1.0 - ciou).mean()


        #print(lbox.item())
        #####
        temp_prd = prd_f.detach().to('cpu').double()



        gt_map = self.create_gt_map(temp_prd[...,0:4], target_f,indices_f, f_idx)
        gt_map = gt_map.to(cfg.DEVICE)


        obj_tf = gt_map[..., 0] == 1
        noobj_tf = gt_map[..., 0] ==0

        # print(prd_f[..., 4][obj_tf].shape)
        # print(ciou.shape)

        #lobj = self.bce_loss(prd_f[..., 4][obj_tf], ciou.detach())
        lobj = self.bce_loss(prd_f[..., 4][obj_tf],gt_map[..., 5][obj_tf])
        # print()
        # print(prd_f[..., 4][obj_tf].sigmoid())
        # print(gt_map[..., 5][obj_tf])

        lnoobj = self.bce_loss(prd_f[..., 4][noobj_tf], gt_map[..., 5][noobj_tf])


        #lbox = self.mse_loss(prd_f[..., 0:4][obj_tf],gt_map[...,1:5][obj_tf] )
        lcls = self.ce_loss(prd_f[..., 5:][obj_tf] ,gt_map[..., 6][obj_tf].long())
        lseg = self.seg_loss(seg, seg_label)
        #print(lbox.item(), lobj.item(), lnoobj.item(), lcls.item())

        return self.weight_box*lbox + self.weight_obj* lobj + self.weight_noobj * lnoobj + lcls + self.weight_seg * lseg
        #return self.weight_box * lbox + self.weight_obj * lobj + self.weight_noobj * lnoobj


    def split_indices(self, indices_f):
        b = indices_f[0]
        a = indices_f[1]
        h = indices_f[2]
        w = indices_f[3]

        return b, a, h, w

    def create_gt_map(self, temp_prd, target_f ,indices_f,  f_idx) :
        temp_xyxy = torch.zeros(4)
        gt_map = torch.zeros(target_f[...,0:7].shape)
        #class_map = torch.zeros(cfg.BATCH_SIZE, cfg.NUM_ANCHOR, cfg.S[f_idx],  cfg.S[f_idx], cfg.NUM_CLASSES)


        b, a, h, w = self.split_indices(indices_f)


        for idx in range(0,len(b)) :

            b_idx = b[idx]
            a_idx = a[idx]
            h_idx = h[idx]
            w_idx = w[idx]

            pcx = cfg.LEN_WIDTH[f_idx] * (w_idx + temp_prd[b_idx, a_idx, h_idx, w_idx, 0]) # Y3
            pcy = cfg.LEN_HEIGHT[f_idx] * (h_idx + temp_prd[b_idx, a_idx, h_idx, w_idx, 1]) # Y3
            # pcx = cfg.LEN_WIDTH[f_idx] * (w_idx + temp_prd[b_idx, a_idx, h_idx, w_idx, 0] * 1.1 - 0.05)#Y4
            # pcy = cfg.LEN_HEIGHT[f_idx] * (h_idx + temp_prd[b_idx, a_idx, h_idx, w_idx, 1] * 1.1 - 0.05) #Y4
            # pcx = cfg.LEN_WIDTH[f_idx] * (w_idx + temp_prd[b_idx, a_idx, h_idx, w_idx, 0] * 2 - 0.5)  # Y4
            # pcy = cfg.LEN_HEIGHT[f_idx] * (h_idx + temp_prd[b_idx, a_idx, h_idx, w_idx, 1] * 2 - 0.5)  # Y4

            #pcw = cfg.ANCHORS[f_idx][a_idx][0] * torch.exp(temp_prd[b_idx, a_idx, h_idx, w_idx, 2])
            pcw = cfg.ANCHORS[f_idx][a_idx][0] * torch.exp(temp_prd[b_idx, a_idx, h_idx, w_idx, 2])
            # YOLOV3
            pch = cfg.ANCHORS[f_idx][a_idx][1] * torch.exp(temp_prd[b_idx, a_idx, h_idx, w_idx, 3]) # YOLOv3

            # pcw = cfg.ANCHORS[f_idx][a_idx][0] * ((temp_prd[b_idx, a_idx, h_idx, w_idx, 2] * 2) ** 2)
            # pch = cfg.ANCHORS[f_idx][a_idx][1] * ((temp_prd[b_idx, a_idx, h_idx, w_idx, 3] * 2) ** 2)

            px1 = pcx - pcw / 2
            py1 = pcy - pch / 2
            px2 = pcx + pcw / 2
            py2 = pcy + pch / 2

            temp_xyxy[0] = px1
            temp_xyxy[1] = py1
            temp_xyxy[2] = px2
            temp_xyxy[3] = py2

            one_iou= self.iou_one2one(temp_xyxy,target_f[b_idx, a_idx, h_idx, w_idx, 1:5])
            #print(one_iou)
            #one_iou = self.ciou_one2one(temp_xyxy, target_f[b_idx, a_idx, h_idx, w_idx, 1:5]).squeeze()

            #print(old_iou, one_iou)

            # print(one_iou)
            # raise NotImplementedError



            cls_idx = int(target_f[b_idx, a_idx, h_idx, w_idx, 9])

            gt_map[b_idx, a_idx, h_idx, w_idx, 0] = 1.
            gt_map[b_idx, a_idx, h_idx, w_idx, 1:5] = target_f[b_idx, a_idx, h_idx, w_idx, 5:9]
            gt_map[b_idx, a_idx, h_idx, w_idx, 5] = one_iou
            gt_map[b_idx, a_idx, h_idx, w_idx, 6] = cls_idx

            #class_map[b_idx, a_idx, h_idx, w_idx, cls_idx] =1.0 # no use





        return gt_map


    def iou_one2one(self, box1, box2):

        inter_x1 = torch.max(box1[0], box2[0])
        inter_y1 = torch.max(box1[1], box2[1])
        inter_x2 = torch.min(box1[2], box2[2])
        inter_y2 = torch.min(box1[3], box2[3])

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)

        inter_area = inter_w * inter_h

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area



        # if iou < 0 or inter_x1 <0 or inter_y1 <0 :
        #     raise NotImplementedError
        #
        # print('box1 :',box1_area)
        # print('box2 :',box2_area)
        # print('inter : ',inter_area)
        # print('iou :',iou)
        #
        # print()

        #raise  NotImplementedError

        return iou


    def ciou_one2one(self,box1, box2, eps= 1e-7):
        with torch.no_grad():
            b1_x1, b1_x2 = box1[0].reshape(1, 1), box1[2].reshape(1, 1)
            b1_y1, b1_y2 = box1[1].reshape(1, 1), box1[3].reshape(1, 1)

            # b2_x1, b2_x2 = box2[:, 0], box2[:, 2]
            # b2_y1, b2_y2 = box2[:, 1], box2[:, 3]

            b2_x1, b2_x2 = box2[0].reshape(1, 1), box2[2].reshape(1, 1)
            b2_y1, b2_y2 = box2[1].reshape(1, 1), box2[3].reshape(1, 1)


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

