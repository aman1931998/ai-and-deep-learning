
# Leaf Detector using Deep Learning

The leaf detector model is created using Tensorpack built upon TensorFlow library. The dataset for this project is manually annotated.<br><br>
[Notebook](https://github.com/aman1931998/ai-and-deep-learning/blob/master/Leaf%20Detector%20using%20Deep%20Learning/Leaf%20Detector.ipynb)

<img class="fit" src="https://raw.githubusercontent.com/aman1931998/ai-and-deep-learning/master/Leaf%20Detector%20using%20Deep%20Learning/model_outputs.png">

## Model
The model used for this project is Mask R-CNN model with Resnet 101 backend neural network. 

## Features

- Uses TensorFlow in the **efficient way** with no extra overhead.
- Focus on **large datasets**.
- **Multi-GPU / multi-node** distributed training, multi-GPU evaluation.
- Cross-GPU BatchNorm (aka **Sync-BN**).
- Group Normalization.
- **Real-time** Tensorboard logging.
- Training from scratch / Transfer Learning.

## Dataset
The dataset used for this model was initially self-annotated using [LabelImg](https://pypi.org/project/labelImg/1.4.0/). It consists of 597 images with 2508 leaf annotations. Later during the course of training, I found another dataset from [roboflow.ai](https://roboflow.ai/) with 1100 more annotated images. 
With augmentation, the dataset size was increased to 11768 images with 40k annotations.
Links for dataset: 
- [COCO formatted/ready](https://drive.google.com/open?id=1l-knrWqArBp5qq6_c0QVlSUV0Us2By3g)
- [Raw data](https://drive.google.com/open?id=1pTZnbBiSbYZmZb_QPL5xx6Eau_e_mPcW)
## Dependencies
+ OpenCV, TensorFlow ≥ 1.6 [conda install]
+ SciPy, Cython [pip install]
+ [pycocotools](https://pypi.org/project/pycocotools/)
+ [tensorpack](https://github.com/tensorpack/tensorpack)
+ Pre-trained [ImageNet ResNet model](http://models.tensorpack.com/#FasterRCNN)

## Structure of Model

### Dataset Structure (COCO) [For training only]
It needs to have the following directory structure:
```
COCO/DIR/
  annotations/
    instances_train201?.json
    instances_val201?.json
  train201?/
    # image files that are mentioned in the corresponding json
  val201?/
    # image files that are mentioned in corresponding json
```
You can use either the 2014 version or the 2017 version of the dataset. The dataset provided by me is following 2017 version.

### File Structure

This is a minimal implementation that simply contains these files:
+ train.py: main training script.
+ predict.py, Leaf Detector.ipynb: main inferencing script.
+ dataset/coco.py: load COCO data to the dataset interface
+ config.py: Data-model configuration file.

## Usage

### Training: 
- Downlaod the dataset using provided link above.
- You can either continue training from [trained weights](https://drive.google.com/open?id=1FBD9hnooE6GiYqDMpwQ1SdAVneF3ZNh_) or start fresh with [ImageNet](http://models.tensorpack.com/FasterRCNN/ImageNet-R101-AlignPadding.npz) weights.
- Place the dataset contents in "coco_format_dataset" folder. 
- Modifications in "config.py":
-DATA.NUM_WORKERS: This value defines the number of pre-processing threads to generate. 
-BACKBONE.WEIGHTS: path to weights file.
-DATA.BASEDIR: path to dataset.
-PREPROC.TRAIN_SHORT_EDGE_SIZE : size of images.
-TRAIN.STEPS_PER_EPOCH: steps per epoch.
- Place the weights file in "train_log" folder.
- Run the following command in the terminal:
```
./train.py --config \
    BACKBONE.WEIGHTS=/path/to/ImageNet-R101-AlignPadding.npz \
    DATA.BASEDIR=/path/to/COCO/DIR \
    [OTHER-ARCHITECTURE-SETTINGS]
```
- Or, if you wish you can update the config.py file and run "train.py" file.

### Inference: 

Note 1: GPU or MKL support is needed.
Note 2: By default, test images and test inferences will be stored in "test_images" and "test_inferences" directories. 

To predict on test images, you have following options:
- **[Recommended]** Run Jupyter Notebook "Leaf Detector.ipynb".
The notebook is self explanatory. Follow the steps provided in it.

- Run "predict.py" in terminal.
To predict on given images:
```
./predict.py --predict input1.jpg input2.jpg --load /path/to/Trained-Model-Checkpoint --config SAME-AS-TRAINING
```
To evaluate the performance of a model on COCO:
```
./predict.py --evaluate output.json --load /path/to/Trained-Model-Checkpoint \
    --config SAME-AS-TRAINING
```

