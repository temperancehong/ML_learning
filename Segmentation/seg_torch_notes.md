The codes are from [github repository](https://github.com/dhruvbird/ml-notebooks/tree/main/pets_segmentation).

Referencing article [Efficient Image Segmentation Using PyTorch](https://towardsdatascience.com/efficient-image-segmentation-using-pytorch-part-3-3534cf04fb89)

## Data Augmentation

Use test set as validation set, and apply data augmentation on the test set. In single training epoch train on the entire data.

## Convolution, Batch Normalization, ReLU, max pooling/unpooling

A typical CNN progressively reduces the input spatial dimensions as layers are stacked. Usually achieved by pooling max or mean values.

A convolution layer has a set of learnable filters that act on small regions in the input to produce a representative output value for each region.

Intuitively, the layers higher up get to “see” a larger region of the input. For example, a 3x3 filter in the second convolution layer operates on the output of the first convolution layer where each cell contains information about the 3x3 sized region in the input. If we assume a convolution operation with stride=1, then the filter in the second layer will “see’’ the 5x5 sized region of the original input.

**Batch normalization** is to normalize each channel independently in the batch input to zero mean and unit variance.

**ReLU**: ensure the non-linearity in the model. Usually followed by a pooling layer to shrink the dimension.

Pooling with stride=2 will transform an input with spatial dimensions (H, W) to (H/2, W/2). Usually max pooling or average pooling.

## SegNet

Explanation reference [SegNet Explained](https://towardsdatascience.com/efficient-image-segmentation-using-pytorch-part-2-bed68cadd7c7).

Encoder section: down-sample to generate features to represent the inputs.

Decoder section: up-sample to create per-pixel classification. 

We can reduce the number of parameters (equivalent to reducint memory) by using Depthwise Separable Convolutions instead of standard convolution


## Small notes

`model.eval()`: `model.eval()` is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn them off during model evaluation, and `.eval()` will do it for you. In addition, the common practice for evaluating/validation is using `torch.no_grad()` in pair with `model.eval()` to turn off gradients computation:

# Depthwise separable convolutions

Replace convolutional layers with depth-wise-separable-convolutions(DSC).

## Evaluation of Trainable Parameters

We have n filters (n kernels). The input dimension is: m * h * w dimension. If the kernels are of size m * dk * dk. If dk=1, then we have the output as n * h *w.

## Depthwise SegNet

Replace each `conv2d` layer by a depthwise convolutional layer that we define. 

The depthwise convoltuion layer is separated into two steps:
- A depthwise grouped convolution, where the number of input channels m is equal to the number of output channels such that each output channel is affected only by a single input channel. In PyTorch, this is called a “grouped” convolution. You can read more about grouped convolutions in PyTorch here.
- A pointwise convolution (filter size=1), which operates like a regular convolution such that each of the n filters operates on all m input channels to produce a single output value

Therefore, the in the view of trainable parameters, there is a 7 times reduction.

Before, for each convolutional layer with `m` in channels and `n` out channels, and `dk` as kernel size, there are `m*dk*dk*n`