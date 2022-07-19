
import Config as cfg

import cv2
import numpy as np
import os
import torch
import torch.utils.data.dataloader
from torch.utils.data import Dataset
import torchvision.transforms as trf

def det_loader(path_data, transform , cvt_type) :
    temp_dataset = Dataset_Det( path_data=path_data,transform=transform,cvt_type = cvt_type)

    temp_data_loader = torch.utils.data.DataLoader(
        temp_dataset, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
        shuffle=True, drop_last=True, pin_memory=True, collate_fn=temp_dataset.custom_collate
    )

    return temp_data_loader


class Dataset_Det(Dataset) :
    def __init__(self, path_data='D:/FAVNet/Data/', transform=None, cvt_type=cv2.COLOR_BGR2RGB):


        self.path_img = path_data + 'X_Image/'
        self.path_det = path_data + 'Y_Detection/'
        self.path_seg = path_data + 'Y_Segmentation/'

        self.img_list = os.listdir(self.path_img)
        self.det_list = os.listdir(self.path_det)
        self.seg_list = os.listdir(self.path_seg)

        self.transform=transform
        self.cvt_type= cvt_type

    def __len__(self):
        return  len(self.img_list)



    def __getitem__(self, idx):
        path_img = self.path_img + self.img_list[idx]
        path_det = self.path_det + self.det_list[idx]
        path_seg = self.path_seg + self.seg_list[idx]
        path_data = [path_img, path_det, path_seg]

        # Image

        image = cv2.imread(path_img)
        image = cv2.cvtColor(image, self.cvt_type)
        # print('image')
        # print(image.shape)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = np.reshape(image,(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 1))

        # Label
        label = []

        #print(path_data[1])
        with open(path_det, 'rt') as f:
            temp_data = f.readlines()

        try :
            for line in temp_data:
                line = line.replace("\n", "").split(" ")

                label.append([
                    float(line[1]),
                    float(line[2]),
                    float(line[3]),
                    float(line[4]),
                    float(line[0])])

        except :
            print(path_det)


        label = torch.Tensor(label)

        # Segn
        mask = cv2.imread(path_seg)
        mask = cv2.cvtColor(mask, self.cvt_type)
        # print('mask')
        # print(mask.shape)
        #print(label)

        transformed = self.transform(image=image, bboxes=label, mask=mask)
        tr_image = transformed['image']
        tr_label = transformed['bboxes']
        tr_label = torch.Tensor(tr_label)
        tr_mask = transformed['mask']

        # print('image')
        # print(tr_mask.shape)
        #
        # print('mask')
        # print(tr_image.shape)
        #
        #
        #
        # print()

        # print(tr_image.shape)
        # print(tr_mask.shape)
        # tf= trf.ToPILImage()
        # img_t= tf(tr_mask)
        #img_t.show()

        # raise NotImplementedError
        #print(tr_label)


        if len(tr_label) == 0 :
            #print('ZERRR!!!')

            image = cfg.NONE_TFM(image=image)['image']


            output = torch.zeros(label.shape)
            # print(output.shape)
            # print(label.shape)

            output[:, 0] = label[:, 4]
            output[:, 1:5] = label[:, 0:4]

            seg_label = self.mask2label(torch.from_numpy(mask))

            #print(image.shape)
            #print(label)
            return image, output, path_data , seg_label

        else :

            output = torch.zeros(tr_label.shape)
            output[:,0] = tr_label[:,4]
            output[:, 1:5] = tr_label[:, 0:4]

            seg_label = self.mask2label(tr_mask)

            return tr_image, output, path_data, seg_label

    def custom_collate(self, batch_data):
        img_data = []

        det_data = []
        path_data= []
        seg_data = []
        for data in batch_data :
            img_data.append(data[0])

            det_data.append(data[1])
            path_data.append(data[2])
            seg_data.append(data[3])

        img_data = torch.stack(img_data,dim=0)
        seg_data = torch.stack(seg_data, dim=0)

        return img_data, det_data , path_data, seg_data



    def mask2label(self, tr_mask):
        SEG_RGB = torch.tensor(cfg.SEG_RGB)
        seg_label = torch.zeros(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)

        for seg_idx in range(0, cfg.SEG_CLASSES):
            temp_tf = tr_mask == SEG_RGB[seg_idx]
            temp_tf = temp_tf[..., 0] * temp_tf[..., 1] * temp_tf[..., 2]

            seg_label[temp_tf] = seg_idx

        return seg_label