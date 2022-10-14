# USOA
Unified model of Semantic segmentation and Object detection for Autonomous driving

Paper : https://lib.dongguk.edu/search/media/url/CAT000001257272

### This description is based on Win10 + Anaconda3

##  Enviroments
####  OS : Win10
####  Manage Syatem : Anaconda3
####  Package Version
  - Python = 3.8.12
  - Pytorch = 1.10.1
  - Albumentation = 1.0.3
  - OpenCV = 4.0.1
  - NVIDIA Driver : 472.47 

#### Train Dataset : BDD100K train set + Argumentation(My Paper)
#### Test Dataset :  BDD100K Validation set
#### Image Size : 640 x 384
#### Object Detection Class : Vehicle(Car + Bus + Truck + Train )
#### Semantic Segmentation Class : Background, Road(Alternative + Direction)

## Model Architecture
![모델구조](https://user-images.githubusercontent.com/60498651/181453260-1a847694-125d-4be1-906e-27591cc5c739.png)


## Inference Video (BDD100K Test)
[![Video Label](http://img.youtube.com/vi/rAvok4emD-8/0.jpg)](https://youtu.be/rAvok4emD-8)

Daytime : https://www.youtube.com/watch?v=rAvok4emD-8




[![Video Label](http://img.youtube.com/vi/m36-rhSQ4cI/0.jpg)](https://youtu.be/m36-rhSQ4cI)

Night : https://www.youtube.com/watch?v=m36-rhSQ4cI


## Performance

|Model|Size|AP(IOU=0.5)|mIoU|FPS(RTX3090)|
|---|---|---|---|---|
|YOLOP|640x384|76.5|91.5|46.51|
|USOA|640x384|76.79|92.57|41.49|

### Paper Link : Preparing...

## Inference Image Example

![00a2e3ca-5c856cde](https://user-images.githubusercontent.com/60498651/179732932-057053b0-2ed8-41e5-a68d-aac92f58b519.jpg) 
![00a2e3ca-5c856cde](https://user-images.githubusercontent.com/60498651/179732960-f5aadc3d-622e-48fe-bb52-c8c42ceb4be4.png)
![00a04f65-af2ab984](https://user-images.githubusercontent.com/60498651/179733177-cc518f4e-0949-4b37-b4b3-c3251a04e25a.jpg) 
![00a04f65-af2ab984](https://user-images.githubusercontent.com/60498651/179733188-b34d3443-66a2-4f79-9ff0-8fc3dbf24b80.png)


## How to Install package
    0. Update Your NVIDIA Graphic Driver 
    1. Open Anaconda3 prompt
    2. conda create -n usoa python=3.8.12
    3. conda activate usoa
    4. conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
    5. conda install -c anaconda cudnn
    6. conda install -c fastai albumentations
 

## Quick Start
####  How to doing?

    0. git clone https://github.com/tmvhs1004/USOA or Download zip, Unzip, Change folder name USOA-main to USOA
    1. Download trained-weight file (File Link : https://drive.google.com/file/d/1oZSQRVQztqOmNqRiVHUy4ZPkl5XkDfLi/view?usp=sharing )
    2. Move trained-weight file to './USOA/Weight/END/ ' Folder 
    3. Open Anaconda Prompt
    4. Cd to USOA folder 
    5. Enter the command 'python test.py'
    6. Waiting for testing time
    7. Inference image is saved in './USOA/Result/output/' folder
    
    
## Training
####  How to doing?

    0. git clone https://github.com/tmvhs1004/USOA or Download zip, Unzip, Change folder name USOA-main to USOA
    1. Download Dataset (File Link) https://drive.google.com/file/d/1K26G7jKbrsHHoiZ6c-7QRUgo7wFfl5M6/view?usp=sharing
    2. Unzip to './USOA/Data/' Folder 
    3. Change code line 115 at Train.py file
      train_set = './Data/Example/' -> train_set = './Data/Train_640x384_refo+w2h2/'
    4. Change hyper-parameter such as 'batch size' in Config.py file
    5. Open Anaconda Prompt
    6. Cd to USOA folder 
    7. Enter the command 'python train.py'
    8. Waiting for training time
    9. The weight file is saved in './USOA/Weight/' folder
   
   

## Testing 
####  How to doing?

    0. git clone https://github.com/tmvhs1004/USOA or Download zip, Unzip, Change folder name USOA-main to USOA
    1. Download Dataset (File Link : https://drive.google.com/file/d/1Zhe58ERgCIkw9yzQRTR9fmGg7U0EzNuu/view?usp=sharing )
    2. Unzip to './USOA/Data/' Folder 
    3. Change code line 145 at Test.py file
      test_set = './Data/Example/' -> test_set = './Data/Test_640x384_refo/'
    4. Download trained-weight file (File Link : https://drive.google.com/file/d/1oZSQRVQztqOmNqRiVHUy4ZPkl5XkDfLi/view?usp=sharing )
    5. Move trained-weight file to './USOA/Weight/END/ ' Folder 
    6. Change hyper-parameter such as 'batch size' in Config.py file
    7. Open Anaconda Prompt
    8. Cd to USOA folder 
    9. Enter the command 'python test.py'
    10. Waiting for testing time
    11. Inference image is saved in './USOA/Result/output/' folder
   
