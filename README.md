# Watson_ChickenProj_JP

This repo stores code to take chicken egg images, apply a prelabled mask, and crop them based on a bounding box formed by a min area rectangle with two normal sides.

## Prerequisites

libraries and packages you'll need that you probably don't already have installed:

```
Open CV ( with support for cv2)
PIL Library
urllib Library
Good understanding of how to interact with json outputs
```

## Running and Using the code

The code can be broken down into two seperate sections:
1. masked_image_generator -used to mask the chicken egg images
   * Vector_getter.py
2. bounding_box_crop_rotate -used to create bounding box and rotate
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
 
 
**Outputs:** a folder called masks to be created. This folder contains all of the newly created images in the same filestructure that the data started as

**Error Handling**: Due to latency, sometimes the urllib request will fail:
```Python
urllib.request.urlretrieve(y, z)
````
To 

```
Give an example
```


```


