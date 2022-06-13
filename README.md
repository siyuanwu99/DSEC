
# A Reproduction of Learning Monocular Dense Depth from Events

### Authors
> [name=Xianzhong Liu]
> [name=Siyuan Wu  | 54888362] 
> [name=Ziyu Ren]

:::warning
Code for this reproduction is available at [<edmundwsy/DSEC>](https://github.com/edmundwsy/DSEC)
:::


[TOC]



---


## Introduction

The work is based on the paper **Learning Monocular Dense Depth from Events** [^7] as a reproduction project, using **DSEC** dataset [^8].

### Event Camera
An event camera, also known as a neuromorphic camera[^1], silicon retina[^2] or dynamic vision sensor[^3], is an imaging sensor that responds to local changes in brightness. Conventional cameras mainly use a shutter to capture images. However, event cameras mainly capture images with each pixels operating independently and asynchronously. It will only report changes of intensity at the pixel level.

Event cameras have a rather high dynamic range, mostly about 120dB[^4], which is about four times of the human eyes'. Event cameras are thus sensors that can provide high-quality visual information even in challenging high-speed scenarios and high dynamic range environments, enabling new application domains for vision-based algorithms. Recently, event cameras have received great interest in object recognition, autonomous vehicles, and robotics[^5].


### Depth Prediction
Depth prediction is the visual ability to perceive the world in three dimensions (3D) and the distance of an object. It has a rather important role in robotics and the automotive industry.

However, while event cameras have appealing properties, they also present unique challenges. Usually, the boundaries of the scene are not clearly prominent due to its working principle. Therefore, it needs to be filled in low contrast regions where no events are triggered, which makes the prediction much challenging.

Early works on event-based depth estimation used multi-view stereo and later Simultaneous Localization and Mapping (SLAM) to build a representation of the environment and therefore derivate metric depth. But these methods either rely on the expensive sensors or use the scale given by available camera poses. Purely vision-based approaches have investigated depth estimation using stereo event cameras, where they rely on maximizing temporal (as opposed to photometric) consistency between a pair of event camera streams to perform disparity and depth estimation.

In this work, they mainly focus on dense, monocular, and metric depth estimation using an event camera, which addresses the aforementioned limitations. **This is the first paper that dense monocular depth is predicted using only events.**

### DSEC dataset
DSEC offers data from a wide-baseline stereo setup of two color frame cameras and two high-resolution monochrome event cameras. In this project, we only use the data from the left event camera as training data and the disparity as ground truth.

![dataset](https://i.imgur.com/WJfVu9k.png)

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

## Training
### Log depth map output
The network is trained to output log depth maps 
$$
\hat{\mathcal{D}}_{k} \in[0,1]^{W \times H}
$$
which have the advantage of representing large depth variations in a compact range, facilitating learning. $k$ refers to the time step of the frame. The metric depth can be recovered by performing the following operations

$$
\hat{\mathcal{D}}_{m, k}=\mathcal{D}_{\max } \exp \left(-\alpha\left(1-\hat{\mathcal{D}}_{k}\right)\right)
$$
where $\mathcal{D}_{\max }$ is the maximum depth range of prediction. $α$ is a parameter chosen, such that a log depth value of 0 maps to minimum observed depth. We take $\mathcal{D}_{\max }=80 m$ and $\alpha = 3.7$, which results in a prediction range of $[2,80]$ meters. 

Therefore, the ground truth disparity is converted firstly to depth
$$
\mathcal{D}_m=\frac{T_{x} f}{-\left[d-\left(c_{x}-c_{x}^{\prime}\right)\right]}
$$

where $d$ is disparity, $T_x$ is the translation distance between left and right cameras(here negative value), $f$ is focal length and $c_{x},c_{x}^{\prime}$ are the coordinates of left and right camera origins on the image (here $c_{x}-c_{x}^{\prime}=0$). After clipping and normalization, the ground truth depth is then converted to log depth

$$
\mathcal{D}_{log}=\frac{log(\frac{\mathcal{D}_m}{\mathcal{D}_{\max }})}{\alpha}+1
$$

Here we use $\mathcal{D}_{log}$ as label for each valid pixel. A pixel is valid if the depth is not infinity, or, equivalently, the disparity is non-zero. 

### Loss
The network is trained to minimize the scale-invariant loss $\mathcal{L}_{k, \mathrm{si}}$ and multi-scale scale-invariant gradient matching loss $\mathcal{L}_{k, \mathrm{grad}}$ at each time step $k$. 
$$
\mathcal{L}_{\mathrm{tot}}=\sum_{k=0}^{L-1} \mathcal{L}_{k, \mathrm{si}}+\lambda \mathcal{L}_{k, \mathrm{grad}}
$$
Given a sequence of ground truth log depth map $\{\mathcal{D}_{k }\}$ , denote the residual $R_k =\hat D_k − D_k$. Then the scale-invariant loss $\mathcal{L}_{k, \mathrm{si}}$ is defined as

$$
\mathcal{L}_{k, \mathrm{si}}=\frac{1}{n} \sum_{\mathbf{u}}\left(\mathcal{R}_{k}(\mathbf{u})\right)^{2}-\frac{1}{n^{2}}\left(\sum_{\mathbf{u}} \mathcal{R}_{k}(\mathbf{u})\right)^{2}
$$

where $n$ is the number of valid ground truth pixels $\mathbf{u}$. This loss function enforces the predicted log depth to be aligned with the ground truth,  irrespective of the absolute global scale [^6]. If the predicted metric depth $\mathcal{D}_m$ is multiplied by scale factor $\beta$ (offset by $\frac{\beta}{\alpha}$ for log depth $\mathcal{D}_{log}$), the loss would still be invariant. 

The multi-scale scale-invariant gradient matching loss $\mathcal{L}_{k, \mathrm{grad}}$ encourages smooth depth changes and enforces sharp depth discontinuities in the depth map prediction.

$$
\mathcal{L}_{k, \operatorname{grad}}=\frac{1}{n} \sum_{s} \sum_{\mathbf{u}}\left|\nabla_{x} \mathcal{R}_{k}^{s}(\mathbf{u})\right|+\left|\nabla_{y} \mathcal{R}_{k}^{s}(\mathbf{u})\right|
$$

where $s$ refers to the different scale. Note that the scale $s$ here refers to the scale of image, not of the scale of depth. Different scale $s$ is obtained by sub-sampling by a factor of 2 on the input and ground truth.

We have chosen the same hyper-parameters as in the paper[^7]. There are totally 4 scales for $\mathcal{L}_{k, \mathrm{grad}}$ (subsampling 4 times) and $λ$ = 0.5. We train with a learning rate of $10^{-4}$ and use the Adam optimizer. 

But due to the limitation on computational resources, we only use a batch size of 4. The training dataset only includes subfiles from DSEC dataset [^8], namely `zurich_city_00_b` and `interlaken_00_c`. 

## Experiments





## Results





## Discussion

1. Suffering from limited computational resources, the network is trained on relatively small amount of data. The total number of images in the training set is 900. This is one of the main reason why our model underforms their original model.



## Conclusion

We have successfully reproduce the paper **Learning Monocular Dense Depth from Events**, enabling depth prediction using monocular event camera data. Though not as good as shown in the paper, the model still achieves fairly good result in the evaluation. We also fill the blank of training code which is not given in the original code website. 


## Reference
[^1]:Li, Hongmin; Liu, Hanchao; Ji, Xiangyang; Li, Guoqi; Shi, Luping (2017). "CIFAR10-DVS: An Event-Stream Dataset for Object Classification". Frontiers in Neuroscience. 11: 309. doi:10.3389/fnins.2017.00309.

[^2]: Sarmadi, Hamid; Muñoz-Salinas, Rafael; Olivares-Mendez, Miguel A.; Medina-Carnicer, Rafael (2021). "Detection of Binary Square Fiducial Markers Using an Event Camera". IEEE Access. 9: 27813–27826. arXiv:2012.06516. doi:10.1109/ACCESS.2021.3058423

[^3]: Liu, Min; Delbruck, Tobi (May 2017). "Block-matching optical flow for dynamic vision sensors: Algorithm and FPGA implementation". 2017 IEEE International Symposium on Circuits and Systems (ISCAS). pp. 1–4. arXiv:1706.05415. doi:10.1109/ISCAS.2017.8050295.

[^4]:Longinotti, Luca. "Product Specifications". iniVation. Retrieved 2019-04-22.

[^5]:Hambling, David. "AI vision could be improved with sensors that mimic human eyes". New Scientist. Retrieved 2021-10-28.

[^6]:Eigen, D., Puhrsch, C., and Fergus, R., “Depth Map Prediction from a Single Image using a Multi-Scale Deep Network”, <i>arXiv e-prints</i>, 2014.

[^7]: J. Hidalgo-Carrió, D. Gehrig and D. Scaramuzza, "Learning Monocular Dense Depth from Events," 2020 International Conference on 3D Vision (3DV), 2020, pp. 534-542, doi: 10.1109/3DV50981.2020.00063.

[^8]: Gehrig, M., Aarents, W., Gehrig, D., and Scaramuzza, D., “DSEC: A Stereo Event Camera Dataset for Driving Scenarios”, <i>arXiv e-prints</i>, 2021.