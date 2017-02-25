# tf-squeezenet
TensorFlow version of SqueezeNet with converted pretrained weights

Current implementation is SqueezeNet v 1.1 (signature pool 1/3/5) without bypasses.

synset_words.txt file is a copt from either caffe or torch tutorials.

Model weights are converted from torch HDF5 model file from https://github.com/rcmalli/keras-squeezenet

Originally, this SqueezeNet was implemented for style transfer, see the original repository here: https://github.com/avoroshilov/neural-style/tree/dev
The style transfer version contains pretrained weights with classifier chopped off, resulting in even smaller file (<3MB).
