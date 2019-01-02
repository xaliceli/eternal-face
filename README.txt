Project: eternal-face
Language: python=2.7.14
Libraries: 
	- dlib 19.1
	- imageio 2.4.1
	- numpy 1.15.4
	- opencv 2.4.13

Running Full Pipeline:

1. Navigate to directory eternal-face/eternal-face where main.py is located.

2. Install any uninstalled packages in environment, e.g.:
	conda install dlib
	conda install imageio

3. Dlib requires a trained facial feature detector, which can be downloaded below.
   Save predictor_68.dat in the 'data' folder.
	https://drive.google.com/open?id=1ckzO-_HdnU-Wx2ISKLvK4ZFAJCn4t9UK

4. Run the following command in the terminal to generate results for image in images/input/set_3 
   (can be replaced with set_1, set_2, or name of any input folder in images/input):
	
	python main.py full set_3 random

5. Interim and final outputs will be generated in the following locations: 
	images/output/set_3/warps: Subregions of input image after undergoing warp, used to generate textures.
	images/output/set_3/textures: Textures generated from a random subset of warped image subregions.
	images/output/set_3/transfers: Intensity from input images transferred onto synthesized textures.
	images/output/set_3/morphs: Videos of texture-transferred images morphing between faces and window sizes.