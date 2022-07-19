from PIL import Image, ImageDraw
import Config as cfg
import torch
import numpy as np
import cv2

def draw_image(path_file, prd_nms) :






    for b_idx, path in enumerate(path_file) :

        temp_xyxy = prd_nms[b_idx]
        #temp_xyxy[...,:4] = temp_xyxy[...,:4] * cfg.IMAGE_SIZE

        temp_xyxy[...,0] = temp_xyxy[...,0] * cfg.IMAGE_WIDTH
        temp_xyxy[..., 1] = temp_xyxy[..., 1] * cfg.IMAGE_HEIGHT
        temp_xyxy[..., 2] = temp_xyxy[..., 2] * cfg.IMAGE_WIDTH
        temp_xyxy[..., 3] = temp_xyxy[..., 3] * cfg.IMAGE_HEIGHT


        #print(temp_xyxy.shape)
        img = Image.open(path[0], mode='r').convert('RGB')
        img = img.resize((cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT),Image.ANTIALIAS)

        draw = ImageDraw.Draw(img)




        for xyxy in temp_xyxy :
            cls_idx = int(xyxy[5])

            draw.rectangle([xyxy[0],xyxy[1], xyxy[2],xyxy[3]], outline=cfg.C_MAP[cls_idx], width=2)
            draw.text((xyxy[0],xyxy[1]-15), cfg.DRV_CLASSES[cls_idx], fill=cfg.C_MAP[cls_idx])


        #img.show()
        img_name = path[0].split('/')[-1]
        img.save('./result/output/'+img_name)




def draw_seg(prd_seg, path_file) :
    _, prd_index = torch.max(prd_seg, dim=1)
    seg_rgb = torch.tensor(cfg.SEG_RGB)

    img_seg = torch.zeros((cfg.BATCH_SIZE,  cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3))

    # print(prd_seg.shape)
    # print(img_seg.shape)

    for cls_idx in range(0, cfg.SEG_CLASSES) :
        seg_tf = prd_index == cls_idx

        img_seg[seg_tf] = seg_rgb[cls_idx].float()


    img_seg = img_seg.int()
    img_seg = img_seg.numpy()
    test_seg = img_seg.astype(np.uint8)



    for b_idx, path_data in enumerate(path_file) :
        seg_name = path_data[2].split('/')[-1]

        image = cv2.cvtColor(test_seg[b_idx], cv2.COLOR_RGB2BGR)

        #print(seg_name)
        cv2.imwrite('./result/output/'+seg_name, image)

