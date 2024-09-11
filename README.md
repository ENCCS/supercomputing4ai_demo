# Supercomputing for AI workshop - Image recognition on HPC

This small example shows how a neural network can be trained and deployed on an HPC resource to perform image recognition
on satellite data. The aim is to recognise water bodies.

## Build the Singularity image

A Singularity image based on the public ROCm-enabled Tensorflow image can be built from the bundled def file with:

```bash
singularity build tensorflow_rocm.sif build_singularity.def
```

## Training the network

A number of images is provided in the repo in the `unet/data` folder. The Unet model can be trained from the root directory of the 
repo as follows:

```bash
singularity exec --bind ./models:/models python models/unet/main.py
```

This will train the model using a GPU and fall back onto CPU backend if a GPU is not available. The script will save the weights under the 
`models/unet/result/models` folder and the results of the training (accuracy, loss, tests) under the `models/unet/result/training` folder.

## Inference

A sample image is available in the `images` folder. Inference can be performed with:

```bash
singularity exec --bind ./models:/models,./images:/images python models/serving/main.py water_body_17.jpg
```

A file will be created under the `images/generated-images` folder with the result.