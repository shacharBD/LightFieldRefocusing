import collections
import os

import imageio
import numpy as np

from utils.image_utils import read_single_image,write_single_image
from utils.interp2 import interp2linear

def shift_views(lightfield_size,m,n,focus,views_dict,views_path,num_of_views):
    """
    Inputs:
    lightfield_size: int, number of lightfield views
    m: int, vertical image dimension
    n: int, horizontal image dimension
    focus: float, focus factor (named 'pixels' in the paper)
    views_dict: dictionary, described for each view its grid location and image name
    views_path: string, path of the images
    num_of_views: int, number of views to shift
    
    This function shifts a given view according to its grid location and focus
    factor, and saves it in the main folder. If you want to feed these views 
    to the model, move them into the shifted_views folder.
    """
    XX,YY = np.meshgrid(np.linspace(0,n-1, n), np.linspace(0,m-1, m)) 
    for i in range(num_of_views):
        curr_view = 'I'+str(i+1)
        coord_x,coord_y = views_dict[curr_view].grid_coords
        image_name = views_dict[curr_view].view_name
        image_path = os.path.join(views_path,image_name)
        curr_image  =  read_single_image(image_path)
        shifted_view = np.zeros([m , n, 3]) 
        shift_x = focus*(coord_y-(np.floor(lightfield_size/2)))
        shift_y = focus*(coord_x-(np.floor(lightfield_size/2)))
        for k in range (3):
            shifted_view[:,:,k] = interp2linear(curr_image[:,:,k],XX+shift_x, YY+shift_y,extrapval=1)
        write_single_image(curr_view+'.png', shifted_view)


lightfield_size = 9
num_of_views = 4
focus = -0.75
main_path = os.path.dirname(os.path.realpath(__file__))
views_path = main_path+"/views/"

#image dimensions [m,n,3]
m = 376
n = 541

View = collections.namedtuple('View', ['grid_coords', 'view_name'])
views_dict = {'I1': View(grid_coords=[4,2], view_name='039.png'),
            'I2': View(grid_coords=[2,4], view_name='023.png'),
            'I3': View(grid_coords=[4,6], view_name='043.png'),
            'I4': View(grid_coords=[6,4], view_name='059.png')}

shift_views(lightfield_size,m,n,focus,views_dict,views_path,num_of_views)
