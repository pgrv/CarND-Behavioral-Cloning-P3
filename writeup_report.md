# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia_model.png "Model Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the Nvidia Net that is described in the classroom:

| Layer         		|     Description	        									| 
|:---------------------:|:-------------------------------------------------------------:| 
| Lambda         		| Image normalization, mean centering, outputs 160x320x3		| 
| Cropping		     	| outputs 65x320x3											 	|
| Convolution 5x5     	| 2x2 stride, outputs 31x158x24, RELU							|
| Convolution 5x5     	| 2x2 stride, outputs 14x77x36, RELU							|
| Convolution 5x5     	| 2x2 stride, outputs 5x37x48, RELU								|
| Convolution 3x3     	| outputs 3x35x64, RELU											|
| Convolution 3x3     	| outputs 1x33x64, RELU											|
| Flatten				| outputs 2112 													|
| Dense					| outputs 120													|
| Dense					| outputs 50													|
| Dense					| outputs 10													|
| Dense					| outputs 1														|


#### 2. Attempts to reduce overfitting in the model

I decided to not modify the net by applying layers like dropout.
I reduced the number of epochs so that the validation lost does not increase.
The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 87-88). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 94).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of the three front cameras as input data.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I made the same steps as described in the classroom.
First I have implemented the pipeline with a base net. I only used a flatten layer and a dense layer.
In the second step I integrated image normalization und mean centering using a lambda layer.
Because the simulation does not chow good results, I used a more powerfull net with the LeNet.
While running the simulator I saw that the vehicle drives to the left and fall of the track. Thats because on the first track we had more left corners than right corners. So I add more datas by flipping every image. So that I had so many right corners as left corners in the input data.
I also used the cameras on the left and right side of the vehicle, regarding a correction factor for non-centered cameras.
I used a cropping layer to remove useless informations like the sky.
In the next step I implemented the Nvidia Net. A more powerfull network.
In the simulation the vehicle could already driving around the first track, without leaving the road.
In the last step I integrated a generator because the amount of data got too large and my laptop runs out of memory.

#### 2. Final Model Architecture

I did not any change on the Nvidia Net. So the final model is the same as described earlier.
The Picture below shows the net but the output shapes are not correct. Therefore take al look at the appopriate model architecture.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track one using center lane driving and one lap while I drived the vehicle off the track so that it could learn what to do if it runs in autonomous mode off the track.
Then I recorded two rounds on the second track.
But the simulation does not got me good results. First I was not sure if it was because of bad training data or because the net was not powerfull enough. So I decided to use the datas that are given in the classroom. With this data set the vehicle drives already around the track one without leaving it.
To get much better results this data set could be expanded by more training data from both tracks. Thats maybe the problem why my vehicle can not drive around the second track and maybe because the vehicle drives not smooth enough.
