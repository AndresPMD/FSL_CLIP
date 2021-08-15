# Few Shot Learning with CLIP
Fast Prototype of a Few Shot Learning approach using CLIP as visual feature extractor.

<a href="url"><img src="project_image/FSL_project.png" align="center" height="500" width="650" ></a>
<p></p>

* To Train the models and visualize results please refer to the file FSL Case Study.ipynb (Jupyter-Notebook).

Steps to reproduce the results:

## Install
Create conda environment

    $ conda create -n clip python=3.6
    
Activate the environment:

    $ conda activate clip  

Install Open AI CLIP dependencies:

    $ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.2
    $ pip install ftfy regex tqdm
    $ pip install git+https://github.com/openai/CLIP.git
    

Check if all dependencies are met comparing to the environment.yml file provided.

## Data

Create the folder ./data/coco_crops_few_shot

Place inside the folder the train and test splits. Inside each split folder there must be folders with the name of the class of the images that it contain.

Create the json files required to train/eval the model:

    $ python preproc_data.py


## Weights

Weights for the pretrained models can be downloaded from:

Fine-tuned ResNet pre-trained in ImageNet:
https://drive.google.com/file/d/1OJAYp39uHxQ7kAWvF4B019z_Ns1fR3F2/view?usp=sharing

Fine-Tuned ViT/16 Image Encoder pre-trained with CLIP:
https://drive.google.com/file/d/1VUDLb_YDkaZMawkONV5q41sutrUCfJSn/view?usp=sharing

Please create a folder ./models
Copy the weights inside this folder.

## Training
To train the pre-trained ViT/16 from CLIP, please run:

    $ python train_clip.py



