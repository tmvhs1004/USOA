# USOA

This description is based on Win10 + Anaconda3

##  Enviroments
  OS : Win10
  Manage Syatem : Anaconda3
  Package Version
    1. Python = 3.8.12
    2. Pytorch = 1.10.1
    3. Albumentation = 1.0.3
    4. OpenCV = 4.0.1
    5. Numpy = 1.22.4
    6. Pillow = 8.4.0



## Training 
  Dataset : BDD100K + Argumentation(My Paper)
  Image Size : 640 x 384
  How to do
    1. Download Dataset
    2. Unzip to Data Folder 
    3. Change code line 115 at Train.py file
      train_set = './Data/Example/' -> train_set = './Data/Train_640x384_refo+w2h2/'
    4. Change hyper-parameter such as 'batch size' in Config.py file
    5. Open Anaconda Prompt(anaconda3)
    6. Cd to USOA folder 
    7. Enter the command 'python train.py'
    8. Waiting for training time
    9. The weight file is saved in './USOA/Weight/' folder

## Testing 
