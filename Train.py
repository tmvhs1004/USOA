import tabnanny

import Config as cfg


import Util.Target as TGT
from Util.Dataset import det_loader

import MODEL.Model as MD
import torch.nn as nn


from Loss.YOLO_LossV2 import YoloLossV2

import time
import torch
import torch.optim as optim
import os

import cv2
import warnings

warnings.filterwarnings('ignore')

def main(model,save_text, train_set, cvt_type) :
    #tgt_name, load_idx, learning_rate = load_first()
    # print(tgt_name)
    # print(load_idx)
    # print(learning_rate)

    # model = model.half()
    # = SANGHYEOP().to(cfg.DEVICE)
    # model.load_state_dict(torch.load('./Temp/BDD_DEV_ALL01_40.pth'))
    model.train()

    optimizer = optim.Adam(
        model.parameters(), lr=cfg.LEARNING_RATE
    )


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.REDUCE_INTERVAL, gamma=0.8)

    loss_fn = YoloLossV2()

    train_loader = det_loader(train_set, cfg.TRAIN_TFM,cvt_type=cvt_type)
    min_loss = 999
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for e_idx in range(0, cfg.NUM_EPOCHS):
        #print(cfg.DEVICE)

        epoch_loss = 0

        # print(len(data_loader))
        now = time.time()

        for b_idx, (x, label, path_file, seg_label) in enumerate(train_loader):
            with torch.cuda.amp.autocast(enabled=True):

            # print(b_idx)
                x = x.to(cfg.DEVICE)
                # x = x.half()
                seg_label = seg_label.to(cfg.DEVICE)
                seg_label = seg_label.long()




                prd1, prd2, prd3, seg1, seg2, seg3 = model(x)

                # print(prd1)
                # print(prd2.shape)


                label_chunk = TGT.label2chunk(label)
                indices, target_map = TGT.chunk2target_map(label_chunk)
                loss = loss_fn(prd1, target_map[0], indices[0], 0, seg1, seg_label) \
                       + loss_fn(prd2, target_map[1], indices[1], 1, seg2, seg_label) \
                       + loss_fn(prd3, target_map[2], indices[2], 2, seg3, seg_label)



            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()

            # loss.backward()
            # optimizer.step()

            epoch_loss += loss.item()

            # raise NotImplementedError

        scheduler.step()
        end = time.time()
        mean_loss = epoch_loss / len(train_loader)

        print('Epcoh : {}, loss : {:.6f}, time : {:.1f}'.format(e_idx + 1,mean_loss , end - now))

        if mean_loss < min_loss :
            min_loss = mean_loss

            torch.save(model.state_dict(), './Weight/{}_{}.pth'.format(save_text, e_idx + 1))

if __name__ == "__main__":
    # BK List


    print('Start training')
    main(
        model=MD.USOA().to(cfg.DEVICE),
        save_text='USOA', # save file name
        train_set='./Data/Example/',
        cvt_type=cv2.COLOR_BGR2RGB
    )


