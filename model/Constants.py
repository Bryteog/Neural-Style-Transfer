import os
import sys
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from Constants import *
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor


class CONFIG:
    IMAGE_WIDTH = 400
    IMAGE_HEIGHT = 300
    CHANNELS = 3
    NOISE_RATIO = 0.5
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    VGG_MODEL = "../pretrained-model/imagenet-vgg-verydeep-19.mat"
    STYLE_IMAGE = "../images/stone_style.jpg"
    CONTENT_IMAGE = "../images/content300.jpg"
    OUTPUT_DIR = "../output/"
    
    
def load_vgg_model(path):
    
    vgg = scipy.io.loadmat(path)
    vgg_layers = vgg['layers']
    

    def _weights(layer, expected_layer_name):
        
        wb = vgg_layers[0][layer][0][0][0][0]

        return W, b

        return W, b


    def _relu(conv2d_layer):
        
        return tf.nn.relu(conv2d_layer)


    def _conv2d(prev_layer, layer, layer_name):
        
        W, b = _weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        
        return tf.nn.conv2d(prev_layer, filter = W, strides = [1, 1, 1, 1], padding = "SAME") + b


    def _conv2d_relu(prev_layer, layer, layer_name):
        
        return _relu(_conv2d(prev_layer,layer, layer_name))


    def _avgpool(prev_layer):
        
        return tf.nn.avg_pool(prev_layer, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")


    graph = {}
    graph['input']    = tf.Variable(np.zeros((1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.CHANNELS)), dtype = float32)
    graph['conv1_1']  = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2']  = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = _conv2d_relu(graph['conv3_1', 12, 'conv3_2'])
    graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])

    return graph


def generate_noise_image(content_image, noise_ratio = CONFIG.NOISE_RATIO):
    
    noise_image = np.random.uniform(-20, 20, (1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.CHANNELS)).astype('float32')
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    
    return input_image


def reshape_and_normalize_image(image):
    
    image = np.reshape(image, ((1, ) + image.shape))
    image = image - CONFIG.MEANS
    
    return image


def save_image(path, image):
    
    image = image + CONFIG.MEANS
    image = np.clip(image[0], 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)
    
    
def content_cost(target):
 
    tf.random.set_seed(1)
    a_C = tf.random.normal([1, 1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random.normal([1, 1, 4, 4, 3], mean=1, stddev=4)
    J_content = target(a_C, a_G)
    J_content_0 = target(a_C, a_C)

    print("J_content = " + str(J_content))

    # Test that it works with symbolic tensors
    ll = tf.keras.layers.Dense(8, activation='relu', input_shape=(1, 4, 4, 3))
    model_tmp = tf.keras.models.Sequential()
    model_tmp.add(ll)
    try:
        target(ll.output, ll.output)
        
    except Exception as inst:
        #print("\n\033[91mDon't use the numpy API inside compute_content_cost\n")
        print(inst)
        
        
def layer_style_cost(target):
 
    tf.random.set_seed(1)
    a_S = tf.random.normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random.normal([1, 4, 4, 3], mean=1, stddev=4)
    J_style_layer_GG = target(a_G, a_G)
    J_style_layer_SG = target(a_S, a_G)

    print("J_style_layer = " + str(J_style_layer_SG))
    
    
def style_mat(target):
 
    tf.random.set_seed(1)
    A = tf.random.normal([3, 2 * 1], mean=1, stddev=4)
    GA = target(A)

    print("GA = \n" + str(GA))

    
def total_costs(target):
 
    J_content = 0.2    
    J_style = 0.8
    J = target(J_content, J_style)

    np.random.seed(1)
    print("J = " + str(target(np.random.uniform(0, 1), np.random.uniform(0, 1))))

    
def train_steps(target, generated_image):
 
    generated_image = tf.Variable(generated_image)

    J1 = target(generated_image)
    print(J1)

    J2 = target(generated_image)
    print(J2)    
