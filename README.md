# Super Resolution for Face Masks
This is an implementation of a super-resolution method using subpixel convolution so that we could take the low-resolution face images and make them high resolution in order to print them and make face masks. This is just for fun so that we could print face masks of the people we love. :) We actually used this and printed face masks from low-resolution images. This does not have to be used for face masks, if you, for some reason, need to transform a low-res face image into high-res, it still works. :)

## Dataset

We used the high resolution version of Celeba dataset in order to train the model only on face data. To construct it, follow the directions in https://github.com/tkarras/progressive_growing_of_gans 

## Super Resolution

We used the sub-pixel convolution from the work of Shi et al., "Real-Time Single Image and Video Super-Resolution Using an Efficient
Sub-Pixel Convolutional Neural Network" (https://arxiv.org/pdf/1609.05158.pdf). 


