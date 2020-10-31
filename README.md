# LightFieldRefocusing
Code for our paper "Deep Sparse Light Field Refocusing", BMVC 2020

Instructions:
1. Install the required dependencies according to the environment.yaml file.
2. Extract the views.zip file into the views/ folder.
3. Run shift_views.py to shift the input views. The code will save the shifted images into the main folder.
4. Move the images into the shifted_views/ folder, and create a sub-folder according to the focus. See the example
of the focus -0.75.
5. Run main_refocus.py which will create a refocused image. The image will be saved under the results/ folder.

Note:
Original code for shifting and refocusing was written in Matlab, and was using an additional resizing step. The current python code
does not include this step and therefore the parameter "focus" should be half of the range that was mentioned in the paper (-0.75 instead of -1.50, etc.). 
