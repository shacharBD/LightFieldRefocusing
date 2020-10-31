import os
import time

import numpy as np
import tensorflow as tf

from utils.cnn_utils import refocusNet
from utils.image_utils import create_data_arrays, read_single_image,write_single_image


def run_refocus(sess,m,n,input_data_path,main_path,focus,crop_size=6):
    """
    Inputs:
    sess: tensorflow session
    m: int, vertical image dimension
    n: int, horizontal image dimension
    input_data_path: string, path of the shited images
    main_path: string, main folder path
    focus: float, focus factor (named 'pixels' in the paper)
    crop_size: int, number of pixels to crop from the refocused image (border effects) 

    This function runs the RefocusNet model on the given views, gets a refocused
    image according to the given focus and saved it in the /results folder.
    """
    start_time = time.time()
    input_focus_path = os.path.join(input_data_path,str(focus))
    input_views_array=create_data_arrays(input_focus_path,m,n)
    feed_dict = {x: input_views_array, is_training: False}
    net_out = sess.run(y_out,feed_dict=feed_dict)
    net_out = net_out.reshape([m,n,3])
    net_out[net_out<0] = 0.0
    net_out[net_out>1] = 1.0
    end_time = time.time()
    time_val = end_time-start_time
    write_single_image(main_path+"/results/"+str(focus)+".png",net_out[crop_size:m-crop_size,crop_size:n-crop_size,:])
    print ('Refocusing process ended, time: {0} seconds'.format(time_val))


main_path = os.path.dirname(os.path.realpath(__file__))
load_path = main_path+'/model/model.ckpt'
input_data_path = main_path+"/shifted_views/"
focus = -0.75
num_views = 4

#image dimensions [m,n]
m = 376
n = 541

tf.reset_default_graph()
# with tf.device('/gpu:0'):
with tf.device('cpu'):
    x = tf.placeholder(tf.float32, shape=[None,None,None,3*num_views])
    is_training = tf.placeholder(tf.bool)
    y_out = refocusNet(x,is_training,num_views)                                      
    saver = tf.train.Saver()    
    config = tf.ConfigProto(device_count = {'GPU': 1},allow_soft_placement=True)
    sess = tf.Session(config=config)
    
saver.restore(sess, load_path)

run_refocus(sess,m,n,input_data_path,main_path,focus)
