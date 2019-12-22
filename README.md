# CS231n Assignments
These are my solutions to the assignments of CS231n offered by the CS Department at Stanford in Spring quarter of 2018. All three assignments have been done on my VM instance of Google Cloud Compute Engine service. The TensorFlow notebooks for assignments 2 and 3 are done on NVIDIA Tesla K80 GPU.

## What is CS231n about?
CS231n (Convolutional Neural Networks for Visual Recognition) is one of the most popular introductory courses on deep learning. Offered every spring since 2015, it is the first intrductory course on deep learning from Dr. Fei-Fei Li's group at Stanford. To give you a better view of the popularity of the course, the first lecture video of CS231n spring 2017 (available [here](https://youtu.be/vT1JzLTH4G4?list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk)) has hit an amazing record of 1 million views on YouTube!<br/>
The cornerstone of CS231n is the notion of gaining a deep understanding of the concepts in deep learning, and implementing them from scratch using Numpy and Python. I encourage you to read an article from Andrej Karpathy, the main originator of this course, in Medium about why this deep understanding matters [here](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b). I also highly recommend anyone who is interested in researching the promising field of deep learning to go through the lecture videos, course notes, tutorials, and most importantly, the assignments of CS231n. I am telling you this based on my own experience and the positive feedback of my friends after doing these assignments. I hope that one day every university in the world has a high-qulaity course on deep learning like CS231n!

## Why going through CS231n assignments?
In my opinion, there two major benefits of doing these assignments are: 1) You create everything from scratch by yourself. you learn how to create deep neural networks in Python from simple linear algebraic operations such as dot prodcut and matrix multiplication. This is an extremeley fun practice, just like when you play with LEGO pieces and construct an intricate design with them! 2) The TAs have designed the pipeline for you and you are assured that the pipeline is efficient. So, what you have to do is to implement the core concepts. But, at the same time, you can learn the pipeline too and use it in you future projects in deep learning.

Here is a breif description about each of the assignments, the topics that they cover, and what you can learn from each one of them:

## Assignment 1. Preliminaries
This assignment is a sort of preculde to working with deep neural networks. The topics of this assignment, such as SVM, K-Nearest Neighbor Model, are mostly covered in machine learning courses. But, implementing them here can give you a better insight into the power of artificial neural networks when you work with them in assignment 2.<br/>
Here is a brief description about different sections of assignment 1:

### 1. KNN
In the first section, you have to train a K-Nearest Neighbor model on a sample of the CIFAR-10 dataset, one of the frequently used dataset for educational purposes. To me, the most exciting part of this section is implementing `*_no_loops` functions, where you have to find a way to use vectorized functions of Numpy instead of for loops in Python. 

<p align="center">
<img src="figures/knn.jpg" alt="drawing" width="250"/>
</p>


### 2. SVM
In this section, you have to train a linear SVM model on the CIFAR-10 dataset. In my opinion, the most important part of this section is the implementation of `svm_loss_vectorized`, where you have to implement hinge loss. Once you code it from scratch in Numpy, the differences between this loss and other losses such as cross-entropy will become crystal clear for you, and you will understand how these differences result in different functionalities and capabilities of SVM compared to other classifiers such as feed forward neural netowrks with cross-entropy loss.

<p align="center">
<img src="figures/svm.jpg" alt="drawing" width="250"/>
</p>



### 3. Softmax Classifier
This part can be regarded one of the most important sections of the assignment. You have to implement softmax + cross-entropy loss, which is used in almost all of the deep neural networks nowadays. You have to be careful with the stability of your implementation! It is worth noting that this stability has been also considered in deep learning frameworks, where using `log softmax` is preferred to using `softmax`.

<p align="center">
<img src="figures/softmax.jpg" alt="drawing" width="250"/>
</p>


### 4. Two-Layer Neural Network
This section is a sort of preclude to assignment 2, where you will work with real deep neural networks. But here, you have to train a tiny neural network with just two layers! Nevertheless, a nice thing about deep learning is that the core ideas hold for deeper neural networks too! Even for a gigantic networks with over 100 layers! :)

<p align="center">
<img src="figures/neural-network.jpg" alt="drawing" width="250"/>
</p>


### 5. Image Features
Instead of training your models on raw pixels, in this part, you have to train your models on the HOG and color histogram feature vectors extracted from the CIFAR-10 images. The good news is that you don't have to implement the feature extraction as it is given to you. The models are also same as previous sections and just the inputs have changed.

<p align="center">
<img src="figures/HOG.jpg" alt="drawing" width="250"/>
</p>


## Assignment 2. Neural Networks
This assignment might be the most important assignment of the course which covers the key ideas in training deep neural networks. In this assignment, you will work with various concepts in deep learning, including different networks, important layers, activation functions, optimization algorithms, regularization techniques, and etc, with both Numpy and deep learning frameworks, such as PyTorch and TensorFlow.<br/>
Here is a brief description of the different sections of assignment 2:

### 1. Fully Connected Networks
Learning how Fully connected Netowrks (FCN) work is the entry point to deep learning. You will find the implementation of the forward pass of fully connected layers pretty straightforward. But, deriving the backpropagation for these layers is super imortant. Neverthelss, it is also one line of code and this is the beauty of deep learning! Simple and short implementations lead to powerful methods that can solve challenging problems!<br/>
In addition to fully connected networks, you will also learn how the different optimization algorithms such as SGD, SGD+Momentum, RMSProp, and Adam work. You will also get familiar with some valuable practical tips for training neural networks --which is particularly useful for your future research in this field--, such as overfitting on a small training set and hyperparameter tuning. 

<p align="center">
<img src="figures/FCN.jpg" alt="drawing" width="350"/>
</p>


### 2. Batch Normalization
Implementing Batch normalization (BN) is tricky, mostly beacause deriving the formula for its backpropagation does not seem straightforward at first glance. However, after deriving the formula, you will see that it can be implemented in 3 or 4 lies of code! You will also learn that batch normalization is sensitive to batch size and some variants of it such as layer normalization have fixed this problem. BN was propsed in 2015 by researchers at Google and is an extremely important component in very deep neural networks as it helps remove the gradient vanishing and exploding problem.

<p align="center">
<img src="figures/BN.jpg" alt="drawing" width="100"/>
</p>


### 3. Dropout
Dropout is a method that that helps neural networks overfit less to the training set and gives some sort of ensemble properties to neural networks. Although it is not backed by a strong theoretical foundation, experimentally, it has helped the models to converge to the local minima faster. Dropout is used in many useful models such as RNN language models and LSTM machine translators.

<p align="center">
<img src="figures/Dropout.jpg" alt="drawing" width="350"/>
</p>



### 4. Convolutional Neural Networks
Now you have reached the super important section of all of the assignments. CNNs are probably the focus of CS231n (the name of the course have been adopted from them!). If you are not careful in implementing the convolutional layers, you will end up with lots of for loops which makes its prohibitively slow! This part shows that how proficient you have become in working with Numpy through doing the previous sections of assigments 2 and 1.

<p align="center">
<img src="figures/CNN.jpg" alt="drawing" width="400"/>
</p>
