# A Reproduction of Learning Monocular Dense Depth from Events


> [name=Xianzhong Liu]
> [name=Siyuan Wu  | 54888362] 
> [name=Ziyu Ren]

:::warning
Code for this reproduction is available at [<edmundwsy/DSEC>](https://github.com/edmundwsy/DSEC)
:::


[TOC]



---


## Introduction
### Event Camera
An event camera, also known as a neuromorphic camera[^1], silicon retina[^2] or dynamic vision sensor[^3], is an imaging sensor that responds to local changes in brightness. Conventional cameras mainly use a shutter to capture images. However, event cameras mainly capture images with each pixels operating independently and asynchronously. It will only report changes of intensity at the pixel level.

Event cameras have a rather high dynamic range, mostly about 120dB[^4], which is about four times of the human eyes'. Event cameras are thus sensors that can provide high-quality visual information even in challenging high-speed scenarios and high dynamic range environments, enabling new application domains for vision-based algorithms. Recently, event cameras have received great interest in object recognition, autonomous vehicles, and robotics[^5].
### Depth Prediction
Depth prediction is the visual ability to perceive the world in three dimensions (3D) and the distance of an object. It has a rather important role in robotics and the automotive industry.

However, while event cameras have appealing properties, they also present unique challenges. Usually, the boundaries of the scene are not clearly prominent due to its working principle. Therefore, it needs to be filled in low contrast regions where no events are triggered, which makes the prediction much challenging.

Early works on event-based depth estimation used multi-view stereo and later Simultaneous Localization and Mapping (SLAM) to build a representation of the environment and therefore derivate metric depth. But these methods either rely on the expensive sensors or use the scale given by available camera poses. Purely vision-based approaches have investigated depth estimation using stereo event cameras, where they rely on maximizing temporal (as opposed to photometric) consistency between a pair of event camera streams to perform disparity and depth estimation.

In this work, they mainly focus on dense, monocular, and metric depth estimation using an event camera, which addresses the aforementioned limitations. **This is the first time that dense monocular depth is predicted using only events.**

### Event data type


## Event Representation




## Network Architecture

![Network structure](https://edmundwsy.github.io/assets/img/network_structure.png)

As shown in the figure above, the network we produced has a recurrent, fully convolutional structure. It can simply be divided as a header, an encoder, a decoder and a predictor.

The header of this network is a 2D convolutional layer followed by Batch Normalization. The kernel size is set to be 5. The activation function is ReLU. It takes 15x640x480 event volumn as input.
The encoder consists of three similar layers with different channel size. Each layer has a 2D convolutional layer and a ConvLSTM layer, which has a LSTM structure with a convolultional gate. The kernel size of convolutional layer is 5, and that of ConvLSTM is selected to be 3.
After the encoder is 2 cascade residual layers with kernel size 3. In the residual layer there are 2 convolutional network with Batch Normalization. The activation function is ReLU. Summation is applied over the skip connection.
The decoder has three similar layers with different output channel size. Each layer consists of an upsampling convolution and a normal convolution with kernel size 5. 
Finally, the network use a predictor to output, which is a depth-wise convolution with kernel size 1. 
This network applies summation over all the skip connections. States from the ConvLSTM will be used for the next event volumn.



## Experiments




## Results



## Discussion



## Conclusion




## Reference
[^1]:Li, Hongmin; Liu, Hanchao; Ji, Xiangyang; Li, Guoqi; Shi, Luping (2017). "CIFAR10-DVS: An Event-Stream Dataset for Object Classification". Frontiers in Neuroscience. 11: 309. doi:10.3389/fnins.2017.00309.

[^2]: Sarmadi, Hamid; Muñoz-Salinas, Rafael; Olivares-Mendez, Miguel A.; Medina-Carnicer, Rafael (2021). "Detection of Binary Square Fiducial Markers Using an Event Camera". IEEE Access. 9: 27813–27826. arXiv:2012.06516. doi:10.1109/ACCESS.2021.3058423

[^3]: Liu, Min; Delbruck, Tobi (May 2017). "Block-matching optical flow for dynamic vision sensors: Algorithm and FPGA implementation". 2017 IEEE International Symposium on Circuits and Systems (ISCAS). pp. 1–4. arXiv:1706.05415. doi:10.1109/ISCAS.2017.8050295.

[^4]:Longinotti, Luca. "Product Specifications". iniVation. Retrieved 2019-04-22.

[^5]:Hambling, David. "AI vision could be improved with sensors that mimic human eyes". New Scientist. Retrieved 2021-10-28.