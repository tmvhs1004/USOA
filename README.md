# USOA
Unified model of Semantic segmentation and Object detection for Autonomous driving


### This description is based on Win10 + Anaconda3

##  Enviroments
####  OS : Win10
####  Manage Syatem : Anaconda3
####  Package Version
  - Python = 3.8.12
  - Pytorch = 1.10.1
  - Albumentation = 1.0.3
  - OpenCV = 4.0.1
  - Numpy = 1.22.4
  - Pillow = 8.4.0

#### Train Dataset : BDD100K train set + Argumentation(My Paper)
#### Test Dataset :  BDD100K Validation set
#### Image Size : 640 x 384
#### Object Detection Class : Vehicle(Car + Bus + Truck + Train )
#### Semantic Segmentation Class : Background, Road(Alternative + Direction)

## Quick Start
####  How to doing?

    0. git clone https://github.com/tmvhs1004/USOA or Download zip
    1. Download trained-weight file (File Link : https://drive.google.com/file/d/1oZSQRVQztqOmNqRiVHUy4ZPkl5XkDfLi/view?usp=sharing )
    2. Move trained-weight file to './USOA/Weight/END/ ' Folder 
    3. Open Anaconda Prompt
    4. Cd to USOA folder 
    5. Enter the command 'python test.py'
    6. Waiting for testing time
    7. Inference image is saved in './USOA/Result/output/' folder
    
    
## Training
####  How to doing?

    0. git clone https://github.com/tmvhs1004/USOA or Download zip
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

    0. git clone https://github.com/tmvhs1004/USOA or Download zip
    1. Download Dataset (File Link : https://drive.google.com/file/d/1Zhe58ERgCIkw9yzQRTR9fmGg7U0EzNuu/view?usp=sharing )
    2. Unzip to './USOA/Data/' Folder 
    3. Change code line 145 at Test.py file
      test_set = './Data/Example/' -> test_set = './Data/Test_640x384_refo/'
    4. Download trained-weight file (File Link : https://drive.google.com/file/d/1oZSQRVQztqOmNqRiVHUy4ZPkl5XkDfLi/view?usp=sharing )
    5. Move trained-weight file to './USOA/Weight/END/ ' Folder 
    6. Change hyper-parameter such as 'batch size' in Config.py file
    7. Open Anaconda Prompt
    8. Cd to USOA folder 
    7. Enter the command 'python test.py'
    8. Waiting for testing time
    9. Inference image is saved in './USOA/Result/output/' folder
   
