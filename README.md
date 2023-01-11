# Face_Detection
This is a work in progress

## Goal 
The goal of this project is to build a face detection system.
Face detection is a big task in Computer Vision, mandatory to achieve some "tasks" such as face recognition.
On this challenge, AI based systems are now the state of the art.

To build the system, 2 approaches are considerated here : 
- via transfer learning
- via fully training a neural network

## Dataset 
We use the WIDERFace dataset, released in 2016. 
The dataset contains 32,203 images and labels 393,703 faces. 


## Part 1 : Transfer Learning
Due to computational ressources constraints, transfer learning is the first approach considerated. 
We use a Resnet ( usually used for classification task ) and get rid of the last layers. 