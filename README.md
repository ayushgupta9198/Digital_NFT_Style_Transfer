# Digital_NFT_Style_Transfer

**Goal: 
**

To visualize and create NFTs based on content and style images respectively..

**Model brief:
**

Model approach is to work with content image and take a style image as a reference and generate results based on the style applied on that content image , this code is currently working on GPU however code is also compatible to run on CPU.

**Vision towards goal:
**

Our mission is to generate uniques art based images for any custom content image

**Research involves around the model:
**

We implemented this model for custom images (both content and style) to visualize.
Model is available in both the versions like GPU & CPU.
Initially the model is available only on GPU functionality but I have edited the model code and made it working on custom images and works fine in both computations
We can change the image size dynamically to any resolution where it is working but as soons as we increase the image size , time for generating result increases.
	
**Implementation towards model:
**

We have set up all the necessary things and models in progression of this model.
As you can check the model workflow , already folders are created , you just need to put the images in the right folder and code will generate results.
In code file renaming is performed in which you just have to put the images in respective folders and it will generate the final file based on your file naming in content and style respectively.
Here there are two folders for output one is “Output Images” in this folder all generated images will be saved on every difference of 1000 steps.
Second folder is “Final_images” . This folder contains the final file for specific content and style images.
We will use this file for further usage. 

**Limitation:**

Model is works on any content or on any style but we need to train the model for specific steps for better results(6000 steps).

If we reduce the number of steps then it will affect the results directly.



