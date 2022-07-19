

import Config as cfg


import Util.Target as TGT
from Util.Dataset import det_loader

import MODEL.Model as MD
import time
import Util.Visual as VIS
import torch
import Util.Detect as DCT
import cv2
import torchvision.transforms as trf

# GPU_NUM = 0
# DEVICE = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(DEVICE)


#DEVICE= cfg.DEVICE




def main(model, weight, test_set, cvt_type) :
    #model = MD.DEV_C3().to(cfg.DEVICE)

    model.load_state_dict(torch.load(weight))
    model = model.half()
    model.eval()

 

    test_loader = det_loader(test_set,  cfg.NONE_TFM, cvt_type)


    #idx_map = CVT.create_idx_map()

    t_total=0

    with torch.no_grad() :
        class_num = torch.zeros(cfg.NUM_CLASSES)

        cls_u = torch.zeros(cfg.SEG_CLASSES)
        cls_i = torch.zeros(cfg.SEG_CLASSES)

        metric_total = []

        prd_all = 0
        data_all = 0

        for b_idx, (x, label, path_file, seg_label) in enumerate(test_loader):
            print('Batch id : {}'.format(b_idx))
            # rint(x.shape)

            x = x.to(cfg.DEVICE)
            x = x.half()
            seg_label= seg_label.to(cfg.DEVICE)


            # if b_idx > 0 :
            #     data_te = time.time()
            #     data_all += data_te - data_ts


            t1 = time.time()
            # prd_seg = model(x)
            prd1, prd2, prd3, seg1, seg2, seg3 = model(x)
            t2 = time.time()
            prd_seg = (seg1 + seg2 + seg3) / 3.

            # print(prd1)

            t_total +=(t2-t1)


            # if b_idx == 999 :
            #
            #     print('inference time : {}'.format(t_total / 1000))
            #
            #
            #     break

            b_i, b_u = DCT.batch_inter_union(prd_seg, seg_label)

            cls_i += b_i
            cls_u += b_u

            # data_ts = time.time()

            label_chunk = TGT.label2chunk(label)
            label_chunk = label_chunk.to(cfg.DEVICE)

            prd1_xyxy = DCT.prd2prd_xyxy(prd1, 0)
            prd2_xyxy = DCT.prd2prd_xyxy(prd2, 1)
            prd3_xyxy = DCT.prd2prd_xyxy(prd3, 2)
            #
            prd_xyxy = torch.cat([prd1_xyxy, prd2_xyxy, prd3_xyxy], dim=0)

            prd_xyxy[:,:4] = torch.clamp(prd_xyxy[:,:4], min=0, max = 1)

            #prd_xyxy = prd3_xyxy

            prd_b, label_b = DCT.an2b(prd_xyxy, label_chunk)
            #print(len(prd_b[0]))
            prd_nms = DCT.nms(prd_b)

            metric_chunk , cls_num_b= DCT.create_metric_chunk(prd_nms, label_b)
            class_num += cls_num_b
            metric_total.append(metric_chunk)
        #
        #
            VIS.draw_seg(prd_seg, path_file)
            VIS.draw_image(path_file, prd_nms)
        #
        #
        #
        metric_total = torch.cat(metric_total)

        total_ap = DCT.total_ap(metric_total, class_num)

        print('GAP : {}'.format(total_ap))

        cls_ap = DCT.calcul_ap(metric_total, class_num)
        # print(mAP : {}.format(cls_ap))
        print('AP(Each Class): {}'.format(cls_ap))
        print('mAP: {}'.format(cls_ap.sum() / cfg.NUM_CLASSES))

        cls_iou = cls_i / cls_u
        miou = cls_iou.sum() / cfg.SEG_CLASSES

        print('mIoU average: {}'.format(miou))
        print('IoU(Each Class) : {}'.format(cls_iou))


if __name__ == "__main__":

    print('Inference Start')
    path_weight = './Weight/END/USOA_50.pth'
    main(
        model=MD.USOA().to(cfg.DEVICE),
        weight=path_weight,
        test_set='./Data/Example/',
        cvt_type = cv2.COLOR_BGR2RGB
    )

