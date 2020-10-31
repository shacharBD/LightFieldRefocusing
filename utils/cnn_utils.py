import tensorflow as tf

"""
These functions are used to build the RefocusNet architecture.
Do not change, or else the model will not be loaded properly.
"""

def layers_upper(x, is_training,count,c_in,c_out):
    w,b = get_filter(count,c_in,c_out)
    a1 = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME') + b
    h1 = tf.nn.relu(a1)
    return h1
      
def layers_lower(x, is_training,count,c_in,c_out):
    w,b = get_filter(count,c_in,c_out)
    a1 = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME') + b
    return a1

def get_filter(count,c_in,c_out):
    w = tf.get_variable('w'+str(count), shape = [3,3,c_in,c_out], initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b = tf.get_variable('b'+str(count), shape = [c_out], initializer=tf.constant_initializer(0.0)) 
    return w,b

def upper_and_lower_layers(x, is_training,param_upper,param_lower):
    output_upper = layers_upper(x,is_training,*param_upper)
    output_lower = layers_lower(x,is_training,*param_lower)
    return output_upper,output_lower

def refocusNet(x,is_training,num_views):                                                                              
    output_upper1,output_lower1 = upper_and_lower_layers(x,is_training,[1,3*num_views,64],[2,3*num_views,12])
    output_upper2,output_lower2 = upper_and_lower_layers(output_upper1,is_training,[3,64,64],[4,64,12])
    output_upper3,output_lower3 = upper_and_lower_layers(output_upper2,is_training,[5,64,64],[6,64,12])
    output_upper4,output_lower4 = upper_and_lower_layers(output_upper3,is_training,[7,64,64],[8,64,12])
    output_upper5,output_lower5 = upper_and_lower_layers(output_upper4,is_training,[9,64,64],[10,64,12])
    output_upper6,output_lower6 = upper_and_lower_layers(output_upper5,is_training,[11,64,64],[12,64,12])
    output_lower7 = layers_lower(output_upper6, is_training,14,64,12)    

    sum_lower = tf.add_n([output_lower1,output_lower2,output_lower3,output_lower4,output_lower5,output_lower6,output_lower7])
    out_lower = layers_lower(sum_lower,is_training,15,12,3)
    average = sum_lower[:,:,:,0:3]+sum_lower[:,:,:,3:6]+sum_lower[:,:,:,6:9]+sum_lower[:,:,:,9:12]
    average = average/num_views
    final_output = out_lower+average
    return final_output
