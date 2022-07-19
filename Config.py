import torch
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import albumentations as A
from albumentations.pytorch import ToTensorV2
# GPU Setting
GPU_NUM = 0 # GPU number to use
DEVICE = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(DEVICE)

# Parameter
NUM_WORKERS = 8 # Number of processes for data loading (default = 0)
BATCH_SIZE = 4 # Batch size for training and inference

#
IMAGE_WIDTH= 640 # Image width
IMAGE_HEIGHT=384 # Image height


# Training parameter
NUM_CLASSES = 1 # Object detection class number
SEG_CLASSES = 2 # Semantic segmentation class number

LEARNING_RATE = 0.0003 # Learning rate for model training
REDUCE_INTERVAL = 5 # Learning rate reduction interval


NUM_EPOCHS = 50 # training max epoch


MAP_IOU_THD = 0.5
NMS_IOU_THD = 0.45 # IOU threshold for NMS
OBJ_THD = 0.5 # confidence THD

# Anchor Size
NUM_ANCHOR = 3
ANCHORS = [
    [[0.0258,0.0395],[0.1167,0.1209],[0.1304,0.2915]],
    [[0.1175,0.4187],[0.2495,0.2474],[0.1432,0.7861]],
    [[0.3355,0.4741],[0.6189,0.2594],[0.3077,0.8726]]
] # Anchor Box size


DRV_CLASSES = [
    'Vehicle'

] # Object detection class


SEG_RGB =[
    [0,0,0], # background
    #[86,211,219],
    [219,94,86] # road

] # Segmentation RGB Value /
# Matching




NONE_TFM = A.Compose([
    A.LongestMaxSize(max_size=IMAGE_WIDTH),
    A.PadIfNeeded(
        min_height=IMAGE_HEIGHT,
        min_width=IMAGE_WIDTH,
        border_mode=cv2.BORDER_CONSTANT,
    ),

    A.RandomGamma(gamma_limit=(60,60),always_apply=True),
    A.Normalize(mean=[0,0,0], std=[1,1,1], max_pixel_value=255),


    ToTensorV2()]
) # Basic transformer (If you don't use BDD100K, Delete A.RandomGamma Line )



TRAIN_TFM = A.Compose([
    A.LongestMaxSize(max_size=IMAGE_WIDTH),

    A.PadIfNeeded(
        min_height=IMAGE_HEIGHT,
        min_width=IMAGE_WIDTH,
        border_mode=cv2.BORDER_CONSTANT,
    ),
    A.RandomGamma(gamma_limit=(60,60),always_apply=True),
    A.GaussNoise(var_limit=(10,50), always_apply=True),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.ChannelShuffle(p=0.5),
    A.Normalize(mean=[0,0,0], std=[1,1,1], max_pixel_value=255),
    ToTensorV2()

], bbox_params=A.BboxParams(format='yolo',min_visibility=0.5, label_fields=[])
)


# Fixed Parameter (Don't Touch)

NUM_FEATURE = 3
S_WIDTH = [IMAGE_WIDTH//8, IMAGE_WIDTH//16, IMAGE_WIDTH//32]
S_HEIGHT = [IMAGE_HEIGHT//8, IMAGE_HEIGHT//16, IMAGE_HEIGHT//32]

LEN_WIDTH = [1 / S_WIDTH[0], 1/S_WIDTH[1], 1/S_WIDTH[2]]
LEN_HEIGHT = [1 / S_HEIGHT[0], 1/ S_HEIGHT[1], 1/S_HEIGHT[2]]





C_MAP = [
    (200, 100, 100),
    (0, 0, 100),
    # (0, 0, 200),
    # (100, 0, 0),
    (200, 0, 0),
    # (0, 100, 0),
    (0, 200, 0),
    (100, 100, 0),
    (100, 200, 0),
    (200, 100, 0),
    (200, 200, 0),
    (100, 0, 100),
    (100, 0, 200),
    (200, 0, 100),
    (200, 0, 200),
    (0, 200, 200),
    (0, 100, 100),
    (100, 100, 100),
    (100, 100, 200),
    (100, 200, 100),
    (100, 200, 200),
    (200, 200, 100),
    (200, 200, 200),

]
