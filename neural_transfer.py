import os
import time
import numpy as np

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave, imread, imresize, fromimage, toimage

import keras.backend as K
from keras.applications import vgg16
from keras.preprocessing.image import load_img, img_to_array

import tensorflow as tf
root_dir = os.path.abspath('.')

base_image_path = os.path.join(root_dir, 'base_image.jpg')
style_image_path = os.path.join(root_dir, 'style_image.jpg')

# set image size
img_rows = 400
img_cols = 400

# set image weights
style_weight = 1
content_weight = 0.025
total_variation_weight = 1

def preprocess_image(image_path):
	img = load_img(image_path, target_size=(img_rows, img_cols))
	img = img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = vgg16.preprocess_input(img)

	return img

def deprocess_image(x):
	x = x.reshape((img_rows, img_cols, 3))
	# x = x.transpose((1, 2, 0))

	# Remove zero centre by mean pixel

	x[:, :, 0] += 103.930
	x[:, :, 1] += 116.779
	x[:, :, 2] += 123.68

	# BGR -> RGB
	x = x[:, :, ::-1]
	x = np.clip(x, 0, 255).astype('uint8')

	return x

# placeholders
base_image = K.variable(preprocess_image(base_image_path))
style_image = K.variable(preprocess_image(style_image_path))

final_image = K.placeholder((1, img_rows, img_cols, 3))

# final_image = tf.convert_to_tensor(final_image)
input_tensor = K.concatenate([base_image, style_image, final_image], axis=0)

model = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet',
	include_top=False)

print('Model loading Done.....')

# model.summary()

# extract names of all layers and their corresponding output
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])


# create loss functions
def content_loss(base, final):
    return K.sum(K.square(final - base))


def gram_matrix(x):
        '''Helper function for style_loss'''
        # features = K.batch_flatten(x)
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        gram = K.dot(features, K.transpose(features))
        return gram
    
def style_loss(style, final):
    S = gram_matrix(style)
    F = gram_matrix(final)
    channels = 3
    size = img_rows * img_cols
    return K.sum(K.square(S - F)) / (4. * (channels ** 2) * (size ** 2))


img_width = img_rows
img_height = img_cols

def total_variation_loss(x):
    a = K.square(x[:, :img_width - 1, :img_height - 1, :] - x[:, 1:, :img_height - 1, :])
    b = K.square(x[:, :img_width - 1, :img_height - 1, :] - x[:, :img_width - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

# set content 
loss = K.variable(0.)
layer_features = outputs_dict['block4_conv2']
base_image_features = layer_features[0, :, :, :]
final_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(base_image_features, final_features)


# set style
feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_features = layer_features[1, :, :, :]
    final_features = layer_features[2, :, :, :]
    sl = style_loss(style_features, final_features)
    loss += (style_weight / len(feature_layers)) * sl
loss += total_variation_weight * total_variation_loss(final_image)


# set gradients
grads = K.gradients(loss, final_image)

# set output
outputs = [loss]

# outputs.append(grads)

outputs += grads

f_outputs = K.function([final_image], outputs)

# helper function to calculate loss and gradients together
def eval_loss_and_grads(x):
    x = x.reshape((1, img_rows, img_cols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values



# helper class
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()


# take input image
x = preprocess_image(base_image_path)

# iterate and optimize
for i in range(1):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # save current generated image
    img = deprocess_image(x.copy())
    fname = '_at_iteration_%d.png' % i
    imsave(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))


