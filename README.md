# tf-squeezenet
TensorFlow version of [SqueezeNet][sqz_arxiv] with converted pretrained weights

Current implementation is SqueezeNet v 1.1 (signature pool 1/3/5) without bypasses.

synset_words.txt file is a copt from either caffe or torch tutorials.

Model weights are converted from keras HDF5 model file from https://github.com/rcmalli/keras-squeezenet

Originally, this SqueezeNet was implemented for style transfer, see the original repository here: https://github.com/avoroshilov/neural-style/tree/dev
The style transfer version contains pretrained weights with classifier chopped off, resulting in even smaller file (<3MB).

## Dependencies
* [TensorFlow](https://www.tensorflow.org/versions/master/get_started/os_setup.html#download-and-setup)
* [NumPy](https://github.com/numpy/numpy/blob/master/INSTALL.rst.txt)
* [SciPy](https://github.com/scipy/scipy/blob/master/INSTALL.rst.txt)
* [Pillow](http://pillow.readthedocs.io/en/3.3.x/installation.html#installation)

[sqz_arxiv]: https://arxiv.org/pdf/1602.07360.pdf
