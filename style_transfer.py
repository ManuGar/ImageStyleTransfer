#!/usr/bin/env python
# coding: utf-8

from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import imageio
import time
import os
import wget

img_height = 400
url = "https://3.bp.blogspot.com/-gG2TK3WUCeE/WEwqlahXgkI/AAAAAAAADLY/SRCcdZn0yeUKDFrTDGgLaVnRHwjQcAabgCLcB/s1600/mariposa.jpg"
filename = wget.download(url)
target_image_path = 'mariposa.jpg'
width, height = load_img(target_image_path).size
img_width = int(width * img_height / height)
combination_image = K.placeholder((1, img_height, img_width, 3))


def preprocess_image(image_path):
    img = load_img(image_path,target_size=(img_height,img_width))
    img = img_to_array(img)
    img = np.expand_dims(img,axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    x[:,:,0] += 103.939
    x[:,:,1] += 116.779
    x[:,:,2] += 123.68
    x = x[:,:,::-1]
    x = np.clip(x,0,255).astype('uint8')
    return x

def content_loss(base,combination):
    return K.sum(K.square(combination-base))

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x,(2,0,1)))
    gram = K.dot(features,K.transpose(features))
    return gram

def style_loss(style,combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height*img_width
    return K.sum(K.square(S-C))/ (4. * (channels ** 2) * (size ** 2))

def total_variation_loss(x):
    a = K.square(x[:,:img_height - 1, :img_width-1,:] - x[:,1:,:img_width-1,:] )
    b = K.square(x[:,:img_height - 1, :img_width-1,:] - x[:,:img_height - 1,1:,:] )
    return K.sum(K.pow(a+b,1.25))

class Evaluator(object):  
    def __init__(self,fetch_loss_and_grads):
        self.loss_value = None
        self.grads_value = None
		self.fetch_loss_and_grads= fetch_loss_and_grads
    
    def loss(self,x):
        assert self.loss_value is None
        x = x.reshape((1,img_height,img_width,3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value
    
    def grads(self,x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

def neural_style_transfer(target_folder_path, style_reference_image_path, output_path):

	# get_ipython().system('wget "https://3.bp.blogspot.com/-gG2TK3WUCeE/WEwqlahXgkI/AAAAAAAADLY/SRCcdZn0yeUKDFrTDGgLaVnRHwjQcAabgCLcB/s1600/mariposa.jpg" -O mariposas.jpg')
	#target_folder_path = 'datasetTransformar'
	#style_reference_image_path = 'datasetOriginal/100.jpg'
	target_image = K.variable(preprocess_image(target_image_path))
	style_reference_image = K.variable(preprocess_image(style_reference_image_path))
	input_tensor = K.concatenate([target_image,style_reference_image,combination_image],axis=0)
	model = vgg19.VGG19(input_tensor=input_tensor,weights='imagenet',include_top=False)
	outputs_dict = dict([(layer.name,layer.output) for layer in model.layers])
	content_layer = 'block5_conv2'
	style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
	total_variation_weight = 1e-4
	style_weight = 1.
	content_weight = 0.025
	loss = K.variable(0.)
	layer_features = outputs_dict[content_layer]
	target_image_features = layer_features[0,:,:,:]
	combination_features = layer_features[2,:,:,:]
	loss += content_weight*content_loss(target_image_features,combination_features)
	for layer_name in style_layers:
		layer_features = outputs_dict[layer_name]
		style_reference_features = layer_features[1,:,:,:]
		combination_features = layer_features[2,:,:,:]
		sl = style_loss(style_reference_features,combination_features)
		loss += (style_weight/len(style_layers)) * sl
	loss += total_variation_weight * total_variation_loss(combination_image)
	grads = K.gradients(loss,combination_image)[0]
	fetch_loss_and_grads = K.function([combination_image],[loss,grads])
	evaluator = Evaluator(fetch_loss_and_grads)
	# result_prefix = 'my_result'
	iterations = 100
	imgs = os.listdir(target_folder_path)
	for im in [target_folder_path+"/"+img for img in imgs]:
		x = preprocess_image(im)
		x = x.flatten()
		for i in range(iterations):
			print('Start of iteration',i)
			start_time = time.time()
			x,min_val,info = fmin_l_bfgs_b(evaluator.loss,x, fprime=evaluator.grads, maxfun=20)
			print('Current loss value:', min_val)
			img = x.copy().reshape((img_height,img_width,3))
			img = deprocess_image(img)
			fname = im[im.rfind('/'):]
			imageio.imwrite(output_path +fname,img)
			print('Image saved as', fname)
			end_time = time.time()
			print('Iteration %d completed in %ds' % (i,end_time - start_time))

#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

#print("Objetivo")
#plt.imshow(load_img(target_image_path,target_size=(img_height,img_width)))

#print("Estilo")
#plt.imshow(load_img(style_reference_image_path,target_size=(img_height,img_width)))

#print("Resultado")
#plt.imshow(img)

