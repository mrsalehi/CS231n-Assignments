# CS231n Assignments
These are my solutions to the assignments of CS231n offered by the CS Department at Stanford in Spring quarter of 2018. All three assignments have been done on my VM instance of Google Cloud Compute Engine service. The TensorFlow notebooks for assignments 2 and 3 are done on NVIDIA Tesla K80 GPU.

## What is CS231n about?
CS231n (Convolutional Neural Networks for Visual Recognition) is one of the most popular introductory courses on deep learning. Offered every spring since 2015, it is the first intrductory course on deep learning from Dr. Fei-Fei Li's group at Stanford. To give you a better view of the popularity of the course, the first lecture video of CS231n spring 2017 (available here ---->link<----) has hit an amazing record of 1 million views on YouTube!<br/>
The cornerstone of CS231n is the notion of gaining a deep understanding of the concepts in deep learning, and implementing them from scratch using Numpy and Python. I encourage you to read an article from Andrej Karpathy, the main originator of this course, in Medium about why this deep understanding matters here --->link<---. I also highly recommend anyone who is interested in researching the promising field of deep learning to go through the lecture videos, course notes, tutorials, and most importantly, the assignments of CS231n. I am telling you this based on my own experience and the positive feedback of my friends after doing these assignments. I hope that one day every university in the world has a high-qulaity course on deep learning like CS231n!

## Why going through CS231n assignments?
In my opinion, there two major benefits of doing these assignments are: 1) You create everything from scratch by yourself. you learn how to create deep neural networks in Python from simple linear algebraic operations such as dot prodcut and matrix multiplication. This is an extremeley fun practice, just like when you play with LEGO pieces and construct an intricate design with them! 2) The TAs have designed the pipeline for you and you are assured that the pipeline is efficient. So, what you have to do is to implement the core concepts. But, at the same time, you can learn the pipeline too and use it in you future projects in deep learning.

Here is a breif description about each of the assignments, the topics that they cover, and what you can learn from each one of them:

## Assignment 1. Preliminaries
This assignment is a sort of preculde to working with deep neural networks. The topics of this assignment, such as SVM, K-Nearest Neighbor Model, are mostly covered in machine learning courses. But, implementing them here can give you a better insight into the power of artificial neural networks when you work with them in assignment 2.<br/>
Here is a brief description about different sections of assignment 1:

### 1. KNN
In the first section, you have to train a K-Nearest Neighbor model on a sample of the CIFAR-10 dataset, one of the widely used dataset for educational purposes. To me, the most exciting part of this section is implementing `*_no_loops` functions, where you have to find a way to use built-in functions of Numpy instead of for loops in Python. 

<p align="center">
<img src="figures/knn.jpg" alt="drawing" width="350"/>
</p>


### 2. SVM
In this section, you have to train a SVM model on the CIFAR-10 dataset. In my opinion, the crucial part of this section is the implementation of `hinge loss`. Once you code it from scratch in Numpy, you will understand how and what are the differences between this loss and other losses such as cross-entropy loss.

<p align="center">
<img src="figures/svm.png" alt="drawing" width="350"/>
</p>



### 3. Softmax Classifier
This can be regarded as the most important section of the assignment. You have to implement softmax + cross entropy loss which is used in almost all of the deep neural networks these days.

<p align="center">
<img src="figures/softmax.jpg" alt="drawing" width="350"/>
</p>


### 4. Two-Layer Neural Network
This section can be considered as a preclude to assignment 2 where you will work with deep neural networks. The core idea is exactly the same except with more layers! :)

<p align="center">
<img src="figures/neural-network.jpg" alt="drawing" width="350"/>
</p>


### 5. Image Features
Instead of training your models on raw pixels, in this part you have to train the models on the HOG and color histogram feature vectors extracted from the CIFAR-10 images. THe models are the same but the inputs have changed.

<p align="center">
<img src="figures/HOG.jpg" alt="drawing" width="350"/>
</p>
