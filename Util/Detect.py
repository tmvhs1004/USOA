import Config as cfg
import torch
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




def D_IOU(box1, box2, eps=1e-7) :
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


    return iou - rho2 / c2





def batch_inter_union(prd_seg, seg_label) :
    _, prd_index= torch.max(prd_seg, dim=1)

    batch_u = torch.zeros(cfg.SEG_CLASSES)
    batch_i = torch.zeros(cfg.SEG_CLASSES)



    for cls_idx in range(0, cfg.SEG_CLASSES) :
        prd_tf = prd_index ==  cls_idx
        gt_tf = seg_label == cls_idx

        inter_tf = prd_tf * gt_tf

        inter_cnt = torch.count_nonzero(inter_tf)

        prd_cnt = torch.count_nonzero(prd_tf)
        gt_cnt = torch.count_nonzero(gt_tf)

        union_cnt = prd_cnt + gt_cnt - inter_cnt


        batch_i[cls_idx] = inter_cnt
        batch_u[cls_idx] = union_cnt


    return batch_i, batch_u




def prd2prd_xyxy(prd, f_idx) :


    prd[..., 0:2] = prd[..., 0:2].sigmoid()
    #prd[..., 2:4] = prd[..., 2:4].sigmoid() # YOLOv5
    prd[..., 4: ] = prd[..., 4:].sigmoid()
    # prd[..., 4] = prd[..., 4].tanh()
    # prd[..., 5:] = prd[..., 5:].sigmoid()


    prd_tf = prd[..., 4] > cfg.OBJ_THD
    prd_obj = prd[prd_tf]

    n_box = len(prd_obj)
    #print(n_box)
    #prd_xyxy = torch.zeros(prd_obj[..., 0:7].shape).to(cfg.DEVICE)
    prd_xyxy = torch.zeros(n_box, 7 ).to(cfg.DEVICE)

    indices = prd_tf.nonzero(as_tuple=False)


    for idx, index in enumerate(indices) :
        b = index[0]
        a = index[1]
        h = index[2]
        w = index[3]

        # cls idx
        cls_idx = torch.argmax(prd_obj[idx, 5:])

        # pxpy > cxcy > xyxy
        #print(f_idx, a)
        #
        pcx = cfg.LEN_WIDTH[f_idx] * (prd_obj[idx, 0] + w)
        pcy = cfg.LEN_HEIGHT[f_idx] * (prd_obj[idx, 1] + h)

        # pcx = cfg.LEN_WIDTH[f_idx] * (prd_obj[idx, 0]*1.1 -0.05 + w)
        # pcy = cfg.LEN_HEIGHT[f_idx] * (prd_obj[idx, 1]*1.1 -0.05 + h) #Y4
        # pcx = cfg.LEN_WIDTH[f_idx] * (prd_obj[idx, 0]*2 -0.5 + w)
        # pcy = cfg.LEN_HEIGHT[f_idx] * (prd_obj[idx, 1]*2 -0.5 + h)



        pcw = cfg.ANCHORS[f_idx][a][0] * torch.exp(prd_obj[idx, 2]) # YOLOv3
        pch = cfg.ANCHORS[f_idx][a][1] * torch.exp(prd_obj[idx, 3]) # YOLOv3

        # pcw = cfg.ANCHORS[f_idx][a][0] * ((prd_obj[idx, 2] * 2) ** 2)
        # pch = cfg.ANCHORS[f_idx][a][1] * ((prd_obj[idx, 3] * 2) ** 2)

        px1 = pcx - pcw / 2
        py1 = pcy - pch / 2
        px2 = pcx + pcw / 2
        py2 = pcy + pch / 2



        prd_xyxy[idx, 0] = px1
        prd_xyxy[idx, 1] = py1
        prd_xyxy[idx, 2] = px2
        prd_xyxy[idx, 3] = py2
        prd_xyxy[idx, 4] = prd_obj[idx, 4]
        prd_xyxy[idx, 5] = cls_idx
        prd_xyxy[idx, 6] = b


    return prd_xyxy

def an2b(prd_xyxy, label_chunk) :
    prd_b=[]
    label_b =[]

    # sort on objectness
    _ , prd_indices = torch.sort(prd_xyxy[:, 4], descending=True)
    prd_xyxy= prd_xyxy[prd_indices, : ]

    for b_idx in range(0, cfg.BATCH_SIZE) :
        label_tf = label_chunk[: ,9 ] == b_idx
        prd_tf = prd_xyxy[:, 6] == b_idx

        prd_temp = prd_xyxy[prd_tf]
        label_temp = label_chunk[label_tf]

        prd_b.append(prd_temp[:,:6])
        label_b.append(label_temp)

    return prd_b, label_b

def nms(prd_b) :

    prd_nms = prd_b

    for b_idx in range(0, cfg.BATCH_SIZE) :

        b_num = len(prd_nms[b_idx])

        #print(b_num)
        for n_idx in range(0, b_num) :

            #iou_map = iou_one2many(prd_nms[b_idx][n_idx] , prd_nms[b_idx])
            iou_map = D_IOU(prd_nms[b_idx][n_idx], prd_nms[b_idx]).reshape(-1)


            # print(test[:10])
            # print(iou_map[:10])
            #
            # print(test[10])
            # print(iou_map[10])
            # raise NotImplementedError
            iou_tf = iou_map > cfg.NMS_IOU_THD
            iou_tf[n_idx] = False # self

            cls_tf = prd_nms[b_idx][:, 5] == prd_nms[b_idx][n_idx, 5]

            last_tf = iou_tf * cls_tf
            prd_nms[b_idx][last_tf,:] =0.0


        sel_tf = prd_nms[b_idx][:, 4] !=0.0

        prd_nms[b_idx] = prd_nms[b_idx][sel_tf, :]

        #print(prd_nms[b_idx])
        #print()



    return prd_nms


def iou_one2many(box_one, box_many) :

    inter_x1 = torch.max(box_one[0], box_many[:,0])
    inter_y1 = torch.max(box_one[1], box_many[:,1])
    inter_x2 = torch.min(box_one[2], box_many[:,2])
    inter_y2 = torch.min(box_one[3], box_many[:,3])

    inter_w =  torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h =  torch.clamp(inter_y2 - inter_y1, min=0.0)

    inter_area = inter_w * inter_h
    box_one_area = (box_one[2] - box_one[0] ) * (box_one[3] - box_one[1])
    box_many_area = (box_many[:,2] - box_many[:,0] ) * (box_many[:,3] - box_many[:,1])

    iou_map = inter_area / (box_one_area + box_many_area - inter_area)


    return iou_map


def create_metric_chunk(prd_nms, label_b) :

    num_class = torch.zeros(cfg.NUM_CLASSES)
    metric_chunk =[]

    for b_idx in range(0, cfg.BATCH_SIZE) :
        label_batch = label_b[b_idx]
        prd_batch = prd_nms[b_idx]

        metric_batch = torch.zeros(len(prd_batch),4)
        metric_batch[:,0] = prd_batch[:,4]
        metric_batch[:, 3] = prd_batch[:, 5]

        label_check = torch.zeros(len(label_batch))

        for l_idx ,l_item in enumerate(label_batch) :
            cls_lb = int(l_item[0])

            num_class[cls_lb] +=1


        #print(prd_batch)
        for p_idx, prd_data in enumerate(prd_batch) :

            iou_map= iou_one2many(prd_data[0:4], label_batch[:, 5:9])
            #print(iou_map.shape)
            cls_idx = prd_data[5]
            cls_tf = label_batch[:,0] != cls_idx

            iou_map[cls_tf] = 0.0

            iou_max = torch.max(iou_map)
            idx_max = int(torch.argmax(iou_map))

            #print(iou_map)
            #print(idx_max)

            if iou_max >= cfg.MAP_IOU_THD :
                metric_batch[p_idx,1] = 1.0

                if label_check[idx_max] == 0.0 :

                    label_check[idx_max] =1.0
                    metric_batch[p_idx, 2] = 1.0




        metric_chunk.append(metric_batch)
        #print(metric_batch)



    metric_chunk = torch.cat(metric_chunk, dim=0)


    #print(metric_chunk.shape)
    #raise NotImplementedError
    return metric_chunk , num_class

def total_ap(metric_total, class_num) :
    total_len = 1/ class_num.sum()

    now_len=0.0

    prd_t = 0.0
    prd_f = 0.0

    _, indices = torch.sort(metric_total[:, 0], descending=True)
    metric_total = metric_total[indices]

    total_ap = 0.0

    for m_idx , m_data in enumerate(metric_total) :

        prd_f +=1
        if m_data[1] == 1.0 :
            prd_t +=1

            if m_data[2] == 1.0 :
                now_len+=total_len

        else :
            total_ap += (now_len * prd_t/ prd_f)
            now_len=0.0



    return total_ap


def calcul_ap(metric_total, class_num) :
    _, indices = torch.sort(metric_total[:,0], descending=True)

    # Class AP
    cls_len =  1/ class_num
    metric_total = metric_total[indices]

    len_rc = torch.zeros(cfg.NUM_CLASSES)

    cls_ap = torch.zeros(cfg.NUM_CLASSES)

    prd_t = torch.zeros(cfg.NUM_CLASSES)
    prd_f = torch.zeros(cfg.NUM_CLASSES)



    for m_idx , m_data in enumerate(metric_total) :
        cls_idx = int(m_data[3])
        now_len = cls_len[cls_idx]


        prd_f[cls_idx] +=1
        if m_data[1] == 1.0 :
            prd_t[cls_idx] +=1

            if m_data[2] == 1.0 :
                len_rc[cls_idx]+=now_len


        else :
            cls_ap[cls_idx] += (len_rc[cls_idx] * prd_t[cls_idx]/ prd_f[cls_idx])
            len_rc[cls_idx]=0.0



    return cls_ap