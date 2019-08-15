# Watson_ChickenProj_JP

This repo stores code to take chicken egg images, apply a prelabled mask, and crop them based on a bounding box formed by a min area rectangle with two normal sides. This is found in folder Egg_Cropping.

This repo also stores the code for training ML models using fastai to predict the sex of chicken eggs before they hatch.

# Egg Cropping

## Prerequisites

Libraries and packages you'll need that you probably don't already have installed:

```
OpenCV (with support for cv2)
PIL Library
urllib Library
Good understanding of how to interact with json outputs
```

## Running and Using the code

The code can be broken down into two seperate sections that a performed in series:
1. masked_image_generator -used to mask the chicken egg images
   * Vector_getter.py
2. bounding_box_crop_rotate -used to create bounding box and crop
   * forloop_bounding_box_crop_rotate_script.py


### masked_image_generator
This code is used to download the masks from label box for each respective file via download links from the labelbox export.json
and apply them to said images. This code requires some setup:
- Must have export.json from labelbox
- The script must be in the same directory as the data set folders
- The data must be segmented exactly how it was uploaded to labelbox where each folder is named such: *./datasetname/*
 >for example, this project had folders named such as ./Clutch1_D12/
 
 Once the code is ran, a dataframe is created from the json outputs. This dataframe is then passed through a for loop that grabs each file and its associated mask, and then applies the mask over the image.
 If for some reason the code is terminated mid loop, the code can be ran again and it will continue from where it left off.
 
 
**Outputs:** A directory called *./masks/* is created. This folder contains all of the newly created images in the same filestructure that the data started as.

**Error Handling**: Due to latency, sometimes the urllib request will fail:
```Python
urllib.request.urlretrieve(y, z)
```
to avoid potential issues, if this happens the code will terminate:
```Python
try:
    urllib.request.urlretrieve(y, z) # access the url to download the file
except:
    print("An error has occured on image " + z)
    sys.exit(1) #stops the code

```
**Notes**: *Due to file permissions issues I encounteder with the urllib library, the root folder containing the script will be used for caching the file as the mask is applied. Do not panic, just delete said files after the script is done.*

### bounding_box_crop_rotate_script
This code is used to crop the newly masked images based on a min area defined by a bounding box with two parellel normal sides.

This code requires some setup:
- The script must be in the same root as *./masks/*
- use only *forloop_bounding_box_crop_rotate_script.py*

once the code is ran, the function is defined and is ran on all images located in *./masks/*

**Outputs:** Once an image is ran through the code, it is saved in its original file location which preserves the orignial data structure.

**Helpful Stuff**: The other script in this section is *bounding_box_crop_rotate_script.py*. This script was originally written to test the many tools found in the open cv library and to get an understanding of how the code works. It contains a few lines at the end of it that allows you to view the bounding rectangle of a specific image in a new window called *contours* which will close immeditaly upon a key press:

```Python
img2 = cv2.resize(img, (1438, 1080))
cv2.imshow("contours", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
*Note that the script will not continue to run until the window is closed. Attempting to run another script and/or not closing the window will cause the python console to crash*
