import Config as cfg
import torch
import math




def label2chunk(label) :

    #print(label)
    label_chunk = []
    for b_idx, b_data in enumerate(label) :

        for l_idx, one_data in enumerate(b_data) :
            temp = torch.zeros(1,10)
            temp[0,:5] = one_data

            x1 =  one_data[1] - one_data[3] /2
            y1 =  one_data[2] - one_data[4] /2
            x2 =  one_data[1] + one_data[3] /2
            y2 =  one_data[2] + one_data[4] /2

            # if x1 ==0 or y1 ==0 or x2 ==0 or y2 ==0 :
            #     print(one_data)
            #     print(x1,y1, x2, y2)
            #     raise NotImplementedError

            b  = b_idx

            temp[0, 5] = torch.clamp(x1, min=0., max=1.)
            temp[0, 6] = torch.clamp(y1, min=0., max=1.)
            temp[0, 7] = torch.clamp(x2, min=0., max=1.)
            temp[0, 8] = torch.clamp(y2, min=0., max=1.)
            temp[0, 9] = b

            label_chunk.append(temp)

    label_chunk = torch.cat(label_chunk, dim=0)

    return label_chunk


def iou_cwch1awah(cwch, awah) :

    awah = torch.tensor(awah)

    inter = torch.min(cwch[0], awah[:,:,0]) * torch.min(cwch[1] , awah[:,:, 1])
    union = cwch[0] * cwch[1] + awah[:,:, 0] * awah[:,:, 1] - inter

    iou = inter/ union

    return iou


def c_iou(box1, box2, eps= 1e-7) : # One Too Many
    with torch.no_grad():
        b2_x1 = torch.zeros((9, 1), requires_grad=False)
        b2_y1 = torch.zeros((9, 1), requires_grad=False)
        b2_x2 = torch.zeros((9, 1), requires_grad=False)
        b2_y2 = torch.zeros((9, 1), requires_grad=False)
        box2 = torch.tensor(box2, requires_grad=False)
        # box1= gt x1y1
        # box2 = anchor cwch

        len_width = torch.tensor(cfg.LEN_WIDTH, requires_grad=False)
        len_height = torch.tensor(cfg.LEN_HEIGHT, requires_grad=False)

        b1_x1, b1_x2 = box1[0].reshape(1,1), box1[2].reshape(1,1)
        b1_y1, b1_y2 = box1[1].reshape(1,1), box1[3].reshape(1,1)




        # print(b1_x1, b1_y1)
        # print(b1_x2, b1_y2)

        # print(box2)
        #h = int(torch.div(one_item[2], cfg.LEN_HEIGHT[f_idx], rounding_mode='floor'))
        w_idx = ((b1_x1 + b1_x2) / 2) // len_width
        h_idx = ((b1_y1 + b1_y2) / 2) // len_height



        for a_idx in range(0, 9) :
            f_idx = a_idx//3
            b2_x1[a_idx, 0] = (w_idx[0, f_idx] + 0.5) * len_width[f_idx] - box2[f_idx, a_idx % 3, 0] / 2
            b2_x2[a_idx, 0] = (w_idx[0, f_idx] + 0.5) * len_width[f_idx] + box2[f_idx, a_idx % 3, 0] / 2
            b2_y1[a_idx, 0] = (h_idx[0, f_idx] + 0.5) * len_height[f_idx] - box2[f_idx, a_idx % 3, 1] / 2
            b2_y2[a_idx, 0] = (h_idx[0, f_idx] + 0.5) * len_height[f_idx] + box2[f_idx, a_idx % 3, 1] / 2




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

        alpha = v / (v - iou + (1 + eps))

        return iou - (rho2 / c2 + v * alpha)







def chunk2target_map(label_chunk) :

    indices= [[[], [], [], []],[[], [], [], []], [[], [], [], []]] # [F][4]

    target_map = [
        torch.zeros(cfg.BATCH_SIZE, cfg.NUM_ANCHOR, cfg.S_HEIGHT[0], cfg.S_WIDTH[0], 10),
        torch.zeros(cfg.BATCH_SIZE, cfg.NUM_ANCHOR, cfg.S_HEIGHT[1], cfg.S_WIDTH[1], 10),
        torch.zeros(cfg.BATCH_SIZE, cfg.NUM_ANCHOR, cfg.S_HEIGHT[2], cfg.S_WIDTH[2], 10)
    ]

    for l_idx , one_item in enumerate(label_chunk) :


        iou = iou_cwch1awah(one_item[3:5], cfg.ANCHORS)
        # ciou = c_iou(one_item[5:9],cfg.ANCHORS).reshape(3,3)

        sort_idx =torch.argmax(iou,dim=1)
        # sort_idx = torch.argmax(ciou, dim=1)

        for f_idx in range(0, cfg.NUM_FEATURE) :

            b = int(one_item[9])
            a = int(sort_idx[f_idx])




            #h = int(one_item[2] // cfg.LEN_HEIGHT[f_idx])
            #w = int(one_item[1] // cfg.LEN_WIDTH[f_idx])
            h = int(torch.div(one_item[2], cfg.LEN_HEIGHT[f_idx], rounding_mode='floor'))
            w = int(torch.div(one_item[1], cfg.LEN_WIDTH[f_idx], rounding_mode='floor'))



            gx = (one_item[1] / cfg.LEN_WIDTH[f_idx]) - w #Y3
            gy = (one_item[2] / cfg.LEN_HEIGHT[f_idx]) - h
            # gx = (one_item[1] / cfg.LEN_WIDTH[f_idx] + 0.05- w) /1.1 #Y4
            # gy = (one_item[2] / cfg.LEN_HEIGHT[f_idx] + 0.05 - h)/1.1
            # gx = (one_item[1] / cfg.LEN_WIDTH[f_idx] + 0.5- w) /2 #Y5
            # gy = (one_item[2] / cfg.LEN_HEIGHT[f_idx] + 0.5 - h)/2

            gw = torch.log(1e-16 + one_item[3] / cfg.ANCHORS[f_idx][a][0]) #YOLOv3
            gh = torch.log(1e-16 + one_item[4] / cfg.ANCHORS[f_idx][a][1]) #YOLOv3
            # gw = torch.sqrt(1e-16 + one_item[3] / cfg.ANCHORS[f_idx][a][0]) / 2
            # gh = torch.sqrt(1e-16 + one_item[4] / cfg.ANCHORS[f_idx][a][1]) / 2

            gx = torch.clamp(gx, min=0.0027)
            gy = torch.clamp(gy, min=0.0027)



            indices[f_idx][0].append(b)
            indices[f_idx][1].append(a)
            indices[f_idx][2].append(h)
            indices[f_idx][3].append(w)

            target_map[f_idx][b,a,h,w][0] = 1.
            target_map[f_idx][b,a,h,w][1:5] = one_item[5:9]
            target_map[f_idx][b, a, h, w][5] = gx
            target_map[f_idx][b, a, h, w][6] = gy
            target_map[f_idx][b, a, h, w][7] = gw
            target_map[f_idx][b, a, h, w][8] = gh
            target_map[f_idx][b, a, h, w][9] = one_item[0]

            for a_idx in range(0, 3) :
                if (a_idx!= sort_idx[f_idx]) and  (iou[f_idx][a_idx]  > 0.5) and  (target_map[f_idx][b, a_idx, h, w][0]==0) :
                    target_map[f_idx][b, a_idx, h, w][0] = -1.


    # print(indices[0])
    # print(target_map[0][0,...,0])
    # raise NotImplementedError

    return indices , target_map


