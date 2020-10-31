import numpy as np
import os
import imageio

def read_single_image(image_path):
    """
    Inputs:
    image_path: string, path to read the image.

    This function gets a path to an image, reads it and returns a float image.
    """
    image = imageio.imread(image_path)
    image = image.astype(np.float64)
    image = image/255
    return image

def write_single_image(path,image):
    """
    Inputs:
    path: string, path to read the image.
    image: numpy array, image to save

    This function gets an image and a path and saves the image to the path.
    """
    image = image*255
    image = image.astype(np.uint8)
    imageio.imwrite(path,image)

def create_data_arrays( input_focus_path,m,n,num_views=4):
    """
    Inputs:
    input_focus_path: string, path to the images
    m: int, vertical image dimension
    n: int, horizontal image dimension
    num_views: number of given views

    This function loads the inputs views from the given path and
    stores them in a numpy array.
    """
    inputListValid = sorted(os.listdir(input_focus_path))
    for i in range(num_views):
        view_name = 'I'+str(i+1)+'.png'
        image = read_single_image(os.path.join(input_focus_path, view_name))
        if i==0:
            input_views_array = image
        else:
            input_views_array = np.concatenate((input_views_array,image), axis=2)
    input_views_array = np.expand_dims(input_views_array,axis=0)            
    return input_views_array
