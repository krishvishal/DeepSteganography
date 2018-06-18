# DeepSteganography
Hiding Images in Plain Sight: Deep Steganography

### Unofficial implementation of NIPS 2017 paper: Hiding Images in Plain Sight: Deep Steganography in Pytorch. 

[link to paper](https://papers.nips.cc/paper/6802-hiding-images-in-plain-sight-deep-steganography)


### Abstract

> Steganography is the practice of concealing a secret message within another,
> ordinary, message. Commonly, steganography is used to unobtrusively hide a small
> message within the noisy regions of a larger image. In this study, we attempt
> to place a full size color image within another image of the same size. Deep
> neural networks are simultaneously trained to create the hiding and revealing
> processes and are designed to specifically work as a pair. The system is trained on
> images drawn randomly from the ImageNet database, and works well on natural
> images from a wide variety of sources. Beyond demonstrating the successful
> application of deep learning to hiding images, we carefully examine how the result
> is achieved and explore extensions. Unlike many popular steganographic methods
> that encode the secret message within the least significant bits of the carrier image,
> our approach compresses and distributes the secret image’s representation across
> all of the available bits.

### Architecture

The architecture consists of a preparation network, hiding network and reveal network. 

![Architecture](https://github.com/krishnavishalv/DeepSteganography/blob/master/images/architecture1.png)

### Error equation

![Error](https://github.com/krishnavishalv/DeepSteganography/blob/master/images/arch2.png)


![Error equation](https://github.com/krishnavishalv/DeepSteganography/blob/master/images/steg_loss.png)


The first error terms backprops only through preparation network and hiding network and the second error backprops through all networks.
