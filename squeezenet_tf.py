# Copyright (c) 2017 Andrey Voroshilov

import os
import tensorflow as tf
import numpy as np
import scipy.io
import time

from PIL import Image

from argparse import ArgumentParser

def imread_resize(path):
    img_orig = scipy.misc.imread(path)
    img = scipy.misc.imresize(img_orig, (227, 227)).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    return img, img_orig.shape

def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)
    
def get_dtype_np():
    return np.float32

def get_dtype_tf():
    return tf.float32
    
# SqueezeNet v1.1 (signature pool 1/3/5)
########################################

def load_net(data_path):
    if not os.path.isfile(data_path):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % data_path)

    weights_raw = scipy.io.loadmat(data_path)
    
    # Converting to needed type
    conv_time = time.time()
    weights = {}
    for name in weights_raw:
        weights[name] = []
        # skipping '__version__', '__header__', '__globals__'
        if name[0:2] != '__':
            kernels, bias = weights_raw[name][0]
            weights[name].append( kernels.astype(get_dtype_np()) )
            weights[name].append( bias.astype(get_dtype_np()) )
    print("Converted network data(%s): %fs" % (get_dtype_np(), time.time() - conv_time))
    
    mean_pixel = np.array([104.006, 116.669, 122.679], dtype=get_dtype_np())
    return weights, mean_pixel

def preprocess(image, mean_pixel):
    swap_img = np.array(image)
    img_out = np.array(swap_img)
    img_out[:, :, 0] = swap_img[:, :, 2]
    img_out[:, :, 2] = swap_img[:, :, 0]
    return img_out - mean_pixel

def unprocess(image, mean_pixel):
    swap_img = np.array(image + mean_pixel)
    img_out = np.array(swap_img)
    img_out[:, :, 0] = swap_img[:, :, 2]
    img_out[:, :, 2] = swap_img[:, :, 0]
    return img_out

def get_weights_biases(preloaded, layer_name):
    weights, biases = preloaded[layer_name]
    biases = biases.reshape(-1)
    return (weights, biases)

def fire_cluster(net, x, preloaded, cluster_name):
    # central - squeeze
    layer_name = cluster_name + '/squeeze1x1'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='VALID')
    x = _act_layer(net, layer_name + '_actv', x)
    
    # left - expand 1x1
    layer_name = cluster_name + '/expand1x1'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x_l = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='VALID')
    x_l = _act_layer(net, layer_name + '_actv', x_l)

    # right - expand 3x3
    layer_name = cluster_name + '/expand3x3'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x_r = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='SAME')
    x_r = _act_layer(net, layer_name + '_actv', x_r)
    
    # concatenate expand 1x1 (left) and expand 3x3 (right)
    x = tf.concat([x_l, x_r], 3)
    net[cluster_name + '/concat_conc'] = x
    
    return x

def net_preloaded(preloaded, input_image, pooling, needs_classifier=False, keep_prob=None):
    net = {}
    cr_time = time.time()

    x = tf.cast(input_image, get_dtype_tf())

    # Feature extractor
    #####################
    
    # conv1 cluster
    layer_name = 'conv1'
    weights, biases = get_weights_biases(preloaded, layer_name)
    x = _conv_layer(net, layer_name + '_conv', x, weights, biases, padding='VALID', stride=(2, 2))
    x = _act_layer(net, layer_name + '_actv', x)
    x = _pool_layer(net, 'pool1_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='VALID')

    # fire2 + fire3 clusters
    x = fire_cluster(net, x, preloaded, cluster_name='fire2')
    fire2_bypass = x
    x = fire_cluster(net, x, preloaded, cluster_name='fire3')
    x = _pool_layer(net, 'pool3_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='VALID')

    # fire4 + fire5 clusters
    x = fire_cluster(net, x, preloaded, cluster_name='fire4')
    fire4_bypass = x
    x = fire_cluster(net, x, preloaded, cluster_name='fire5')
    x = _pool_layer(net, 'pool5_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='VALID')

    # remainder (no pooling)
    x = fire_cluster(net, x, preloaded, cluster_name='fire6')
    fire6_bypass = x
    x = fire_cluster(net, x, preloaded, cluster_name='fire7')
    x = fire_cluster(net, x, preloaded, cluster_name='fire8')
    x = fire_cluster(net, x, preloaded, cluster_name='fire9')
    
    # Classifier
    #####################
    if needs_classifier == True:
        # Dropout [use value of 50% when training]
        x = tf.nn.dropout(x, keep_prob)
    
        # Fixed global avg pool/softmax classifier:
        # [227, 227, 3] -> 1000 classes
        layer_name = 'conv10'
        weights, biases = get_weights_biases(preloaded, layer_name)
        x = _conv_layer(net, layer_name + '_conv', x, weights, biases)
        x = _act_layer(net, layer_name + '_actv', x)
        
        # Global Average Pooling
        x = tf.nn.avg_pool(x, ksize=(1, 13, 13, 1), strides=(1, 1, 1, 1), padding='VALID')
        net['classifier_pool'] = x
        
        x = tf.nn.softmax(x)
        net['classifier_actv'] = x
    
    print("Network instance created: %fs" % (time.time() - cr_time))
   
    return net
    
def _conv_layer(net, name, input, weights, bias, padding='SAME', stride=(1, 1)):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, stride[0], stride[1], 1),
            padding=padding)
    x = tf.nn.bias_add(conv, bias)
    net[name] = x
    return x

def _act_layer(net, name, input):
    x = tf.nn.relu(input)
    net[name] = x
    return x
    
def _pool_layer(net, name, input, pooling, size=(2, 2), stride=(3, 3), padding='SAME'):
    if pooling == 'avg':
        x = tf.nn.avg_pool(input, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                padding=padding)
    else:
        x = tf.nn.max_pool(input, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                padding=padding)
    net[name] = x
    return x

def build_parser():
    ps = ArgumentParser()
    ps.add_argument('--in',             dest='input', help='input file', metavar='INPUT', required=True)
    ps.add_argument('--fool',           dest='fool', type=int, help='if image needs to be altered to fool the network classification (argument - class number)', metavar='FOOL')
    return ps

def main():
    import time

    parser = build_parser()
    options = parser.parse_args()

    # Loading image
    img_content, orig_shape = imread_resize(options.input)
    img_content_shape = (1,) + img_content.shape

    # Loading ImageNet classes info
    classes = []
    with open('synset_words.txt', 'r') as classes_file:
        classes = classes_file.read().splitlines()

    # Loading network
    data, sqz_mean = load_net('sqz_full.mat')

    config = tf.ConfigProto(log_device_placement = False)
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'

    g = tf.Graph()
    
    # 1st pass - simple classification
    with g.as_default(), tf.Session(config=config) as sess:
        # Building network
        image = tf.placeholder(dtype=get_dtype_tf(), shape=img_content_shape, name="image_placeholder")
        keep_prob = tf.placeholder(get_dtype_tf())
        sqznet = net_preloaded(data, image, 'max', True, keep_prob)

        # Classifying
        sqznet_results = sqznet['classifier_actv'].eval(feed_dict={image: [preprocess(img_content, sqz_mean)], keep_prob: 1.})[0][0][0]

        # Outputting result
        sqz_class = np.argmax(sqznet_results)
        print("\nclass: [%d] '%s' with %5.2f%% confidence" % (sqz_class, classes[sqz_class], sqznet_results[sqz_class] * 100))

    if options.fool is not None:
        target_class = options.fool
        if target_class >= 1000 or target_class < 0:
            target_class = 0
        print("target class: [%d] '%s' with %5.2f%% confidence\n" % (target_class, classes[target_class], sqznet_results[target_class] * 100))

        iterations = 200
        
        # 2nd pass - producing image that fools the classifier
        with g.as_default(), tf.Session(config=config) as sess:
            img_preprocess = preprocess(img_content, sqz_mean)
            image_fooling = tf.Variable(np.expand_dims(img_preprocess, axis=0), dtype=get_dtype_tf())
            keep_prob = tf.placeholder(get_dtype_tf())
            sqznet = net_preloaded(data, image_fooling, 'max', True, keep_prob)
            
            target = np.empty((1000))
            target.fill(-1)
            target[target_class] = 1
            fooling_loss = tf.nn.l2_loss(tf.constant(target, dtype=tf.float32) - sqznet['classifier_actv'])
            
            train_step = tf.train.AdamOptimizer(1e0, 0.9, 0.999, 1e-8).minimize(fooling_loss)
            
            sess.run(tf.global_variables_initializer())
            print('Fooling started..')
            iter_cnt = 0
            for i in range(iterations):
                print('Iteration %4d/%4d, loss: %f' % (iter_cnt, iterations, fooling_loss.eval(feed_dict={keep_prob: 1.})))
                iter_cnt += 1
                train_step.run(feed_dict={keep_prob: 1.})
            
            fooled = image_fooling.eval()
            fooled = scipy.misc.imresize(unprocess(fooled.reshape(fooled.shape[1:]), sqz_mean), orig_shape)
            
            path, extension = os.path.splitext(options.input)
            fooled_filename = path + '_fool.png'
            
            print("\nSaving fooling image as '%s'\n" % fooled_filename)
            imsave(fooled_filename, fooled)
            
        # 3rd pass - classifying fooling image
        with g.as_default(), tf.Session(config=config) as sess:
            fooled_resize = scipy.misc.imresize(fooled, (227, 227))
        
            # Building network
            image = tf.placeholder(dtype=get_dtype_tf(), shape=img_content_shape, name="image_placeholder")
            keep_prob = tf.placeholder(get_dtype_tf())
            sqznet = net_preloaded(data, image, 'max', True, keep_prob)

            # Classifying
            sqznet_results = sqznet['classifier_actv'].eval(feed_dict={image: [preprocess(fooled_resize, sqz_mean)], keep_prob: 1.})[0][0][0]

        sqz_class = np.argmax(sqznet_results)
        print("\nclass: [%d] '%s' with %5.2f%% confidence" % (sqz_class, classes[sqz_class], sqznet_results[sqz_class] * 100))
        print("target class: [%d] '%s' with %5.2f%% confidence" % (target_class, classes[target_class], sqznet_results[target_class] * 100))
        
if __name__ == '__main__':
    main()