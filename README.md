<h3> Object detection using YOLO V4 </h3> <hr>

<img src="/Yolo V4 Object Detection/Images/1.jpg" width="700" height="400">

classification used to find out the label of the image and localization gives an rectangle box after finding the label and Detection used for tracting or detecting the labels in images or in videos .
For This object detection I am using YOLOV4 architecture with darknet and OpenCV Frame work <hr>
For this purpose I used an model where it developed on 80 labels and data set called coco dataset . 

<img src="/Yolo V4 Object Detection/Images/traffic.webp" width="700" height="400">
<hr>

The above image I converted into Yolo formats where we need to mention mid point of an images height and width and X ordinates and y_cordinates and label for the images 

<img src="/Yolo V4 Object Detection/Images/5.png" width="700" height="400">
In the place of image path we can provide an video path for detecting . The labelled data will be availabel in data folder above 
<hr>

<h2> YOLO means (You only look once) </h2><hr>
The YOLO algorithm was developed in 2015, and it involves one neuron that is trained and takes an image as input and gives a prediction of a bounding box and the class labels as output.
<hr> 

<h2> yolo v4 </h2><hr>
<img src="/Yolo V4 Object Detection/Images/v4.png" width="700" height="400"><hr>
Yolov4 is an improvement on the Yolov3 algorithm by having an improvement in the mean average precision(mAP) by as much as 10% and the number of frames per second by 12%. The Yolov4 architecture has 4 distinct blocks as shown in the image above, The backbone, the neck, the dense prediction, and the sparse prediction. 

<img src="/Yolo V4 Object Detection/Images/vi.gif" ><hr>

<hr> <h2> How to create an bounding boxes for lables inside images or videos  </h2>
Just we can find the label using normal CNN method and when there is any label just find the cordinates of the label and shape of it from mid location . In this way we can find out the deetction for images and each frame of a video 
<img src="/Yolo V4 Object Detection/Images/i.png" width="700" height="400"><hr>

The cordinates for the labels inside the videos are 
<img src="/Yolo V4 Object Detection/Images/j.png" width="700" height="400"><hr>
the Line says there are labels in the and finding out the cordinates <hr>

The Yolov3 and Yolov4 algorithms are both excellent at object detection, but as I pointed out earlier, there are several other algorithms for object detection. Below are the results obtained using yolov3 and yolov4 on the coco dataset for object detection, and some other detection algorithms.
<hr>
<img src="/Yolo V4 Object Detection/Images/c.jpg" width="700" height="400">

<hr>
<h3> Custom Object Detection <h3>
<hr>
After making use of pretrained weights from coc dataset . Trained own object detection model which is used to find whether the person wearing the mask or not . For making an custom object detection I used data around 15k Images . where 7k are from masked people and 7k are without mask . For developing this model used yolov4 custom configuration file which we can get from source = '!git clone https://github.com/AlexeyAB/darknet' .
we must make sure that taking the permisson of GPu which supports cuda version . Because darknet framework were trained on Nvdia cuda Library . for making [!make] file changes follow 
 %cd '/content/drive/MyDrive/yolo/darknet'
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
!sed -i 's/LIBSO=0/LIBSO=1/' Makefile
  
<hr>
cfg file chages 
<hr> 
<img src="/Yolo V4 Object Detection/Images/img10.png" > 
  
                         1 . change line batch to batch=64
                         2 . change line subdivisions to subdivisions=16
                         3 . set network size width=416 height=416 or any value multiple of 32
                         4 . change line max_batches to (classes*2000 but not less than the number of training images and not less than 6000), f.e. max_batches=6000 if you train
 for 3 classes
                         5 . change line steps to 80% and 90% of max_batches, f.e. steps=4800,5400

 <hr>
 
 <h3> What are subdivisions </h3>
 
 <hr>
What are subdivisions? It means how many mini-batches you split your batch in. Batch=64 -> loading 64 images for one iteration. Subdivision=8 -> Split batch into 8 mini-batches so 64/8 = 8 images per mini-batch and these 8 images are sent to the GPU for processing. This process will be performed 8 times until the batch is completed and a new iteration will start with 64 new images. If you are using a GPU where the RAM is low, set a higher value for subdivisions ( 32 or 64). This will obviously take longer to train since we are reducing the number of images being loaded and also the number of mini-batches. If you have a GPU with good RAM, set a lower value for subdivisions (16 or 8). This will speed up the training process as this loads more images per iteration.

<hr> Results </hr>
<h3>for single object detection </h3>
 <img src="/Yolo V4 Object Detection/Images/video2.gif" > 
<hr>
 
 <h3> For Multiple object detection </h3>
 <img src="/Yolo V4 Object Detection/Images/yolov4-mask.gif" > 
 
 <hr> <h3> 10 Labels custom object detection </h3> <hr>
The YOLO implementations are amazing tools that can be used to start detecting common objects in images and videos. However there are many cases,when the object which we want to detect are not part of the popular dataset. In such cases we need to create our own training set and execute our own training. Vehicle and its License plate detection are one such cases. I have used yolov4 to detect the desired classes . 
 
| Vehicle       | Label Id      |
| ------------- |:-------------:|
| Car           |       0       |
| Truck         |       1       |
| Bus           |       2       |
| Motorcycle    |       3       |
| Auto          |       4       |
| CarLP         |       5       |
| TruckLP       |       6       |
| BusLP         |       7       |
| MotorcycleLP  |       8       |
| AutoLP        |       9       |
*License Plate(LP)
 
Later splitted all the dataset in training and validation set and stored the path of all the images in file named train.txt and valid.txt

# Configuring Files
Yolov4 needs certain specific files to know how and what to train.
1. obj.data
2. obj.names
3. obj.cfg

## obj.data
This basically says that we are training 10 classes, what the train and validation files are and which file contains the name of object we want to detect.During training save the weight in backup folder.
```
classes = 10
train = train.txt
valid = test.txt
names = obj.names
backup = backup
```
<hr>
<img src="/Yolo V4 Object Detection/Images/truck.png" >  
<hr> 
## obj.names
Every new cateogry must be in new line and its category number be same what we have used at the time of annotating data.
```
Car
Truck
Bus
Motorcycle
Auto
CarLP
TruckLP
BusLP
MotorcycleLP
AutoLP
```
## obj.cfg

 Just copied the yolov4.cfg files and made few changes in it.
 * In line 3, set `batch=24` to use 24 images for every training step.
 * In line 4, set `subdivisions=8` to subdivide the batch by 8 to speed up the training process.
 * In line 127, set `filters=(classes + 5)*3`, e.g `filter=45`.
 * In line 135, set `classes=10`, number of custom classes.
 * In line 171, set `filters=(classes + 5)*3`, e.g `filter=45`.
 * In line 177, set `classes=10`, number of custom classes.

<img src="/Yolo V4 Object Detection/Images/auto.png" > 


 
