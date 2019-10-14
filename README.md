## Project: Build a Traffic Sign Recognition Program

Overview
---
In this project, a convolutional neural network is utilized to classify traffic signs. The CNN is trained and validated so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).


[//]: # (Image References)

[image1]: ./examples/Img1.jpg "1"
[image2]: ./examples/Img2.jpg "2"
[image3]: ./examples/Img3.jpg "3"
[image4]: ./examples/Img4.jpg "4"
[image5]: ./examples/Img5.jpg "5"
[image6]: ./examples/.LabelDistribution.png "6"
[image7]: ./examples/dataExample.png "7"


![1][image1]    | ![2][image2] | ![3][image3] | ![4][image4] | ![5][image5]  |

Repository contents
---

*Traffic_Sign_Classifier.ipynb* contains a Jupyter notebook with the project code and comments to be run in a browser

*trafficSignClassifier.py* contains the same project code to be run on a local machine

*data* folder containing pickled test, validation and training data from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

*examples* folder containing visualization images



The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images



Dataset Summary and Exploration
---

![7][image7]

I used the numpy library to calculate summary statistics of the traffic signs data set:
* The size of training set is 34799 Images
* The size of the validation set is 4410 Images
* The size of test set is 12630 Images
* The shape of a traffic sign image is 32 x 32 pixels with 3 color channels (RGB)
* The number of unique classes/labels in the data set is 43

![6][image6]


Model Architecture
---

The model used in this project is based on the LeNet architecture. It takes 32 x 32 pixel images with a
single color channel as an input. The model consists of a sequence of two convolutional and maxpooling layers followed by a flattening step and three fully connected layers. All layers are followed
by a RELU activation function. After the first and second fully connected layer a 50% probability
dropout layer is applied. The output of the model is a one dimensional 43 logits long array
representing the 43 traffic sign labels.

* All convolutional layers use a `[1, 2, 2, 1]` stride
* All pooling layers use a `[1, 2, 2, 1]` stride and kernel size
* All weights are initialized with a truncated normal distribution (mu = 0, sigma = 0.1)
* All biases are initialized as zeros


Layer                    | Description                                |
:-----------------------:|:------------------------------------------:|
Convolutional Layer      | Input = 32 x 32 x 1, Output = 28 x 28 x 6  |
Activation               | RELU function                              |
Pooling                  | Max-Pooling , Output = 14 x 14 x 6         |
Convolutional Layer      | Output = 10 x 10 x 16                      |
Activation               | RELU function                              |
Pooling                  | Max-Pooling, Output = 5 x 5 x 16           |
Flatten                  | Output = 400                               |
Fully Connected Layer    | Output = 120                               |
Activation               | RELU function                              |
Dropout Layer            | Keep Probability = 0.5                     |
Fully Connected Layer    | Output = 84                                |
Activation               | RELU function                              |
Dropout Layer            | Keep Probability = 0.5                     |
Fully Connected Layer    | Output = 43                                |


Training Summary
---

To train the model, I used an Adam optimizer which is known to giver slightly better results than
gradient descent in this type of problem.

*HYPERPARAMETERS*:
* Batch size: 128
* Epochs: 25
* Learning rate : 0.0009
* Keep Prob : 0.5
* Mu: 0
* Sigma: 0.1

Validation summary
---
My final model results were:
* validation set accuracy of 0.940
* test set accuracy of 0.925
* example set accuracy 1.00

I chose to use the well known LeNet architecture and slightly modified it.
It is a fairly straight forward architecture known to work well for classification problems like these. My first attempt using LeNet
only gave me a validation accuracy of around 70%. I decided to add two dropout layers after the first
fully connected layers with a keep probability of 0.5. This alone pushed my validation accuracy to
around 90%. I then fine tuned the hyperparameters to get to a validation accuracy greater than 93%.
I was able to get the best results by using a low learning rate of 0.0009 and more Epochs than in the
standard architecture. I ended up using 25 in my final submission. I also tried to tune the keep
probability of the dropout layers but 0.5 seemed to work best.
Certainly it is possible to reach a accuracy higher than mine. Possible ideas to achieve this would be
to implement L2-normalization, more sophisticated preprocessing or generating additional training
data. Adding another fully connected layer could also be taken into consideration.
