# Bachelor project 2023
# Data imputation on microscopic images of boar spermatozoa


## PCA
patch_manager.py: Class used for managing work with patches

reconstructor.py: Class used for reconstruction of 3D images

patch_scale_experiment.py: Used for experiment that shows the corelation between patch size and amount of principal components. (Old design, but works) 

PCA_recon_final.ipynb: Used for reconstructing using PCA. Results are shown in report 

simple_experiment: Used for simple 3x3 PCA experiment from section 5.1.5

## NeuralNetworks

Image_Inpainting_Autoencoder_Decoder_v2_0.ipynb: Used for reconstruction with neural network

data-z.py: Used to prepare data for use in neural network

## motion

compute_angle.py: Used to display motion and rotation. Also computes the rotation of the midpiece

skeletonize.py: Used for skeletonizing and projections using PCA

tif_separator.py: Used to separate .tif files where all layers and timesteps are saved under same variable. Outputs .tif files where each file is a timestep.

(Skeletonize cells and save output -> separate .tif files into timesteps -> compute angles)


## Misc
confusion_calculation.py: Used to calculate confusion matrix of segmentations and groundtruths

swap_view.py: Rotates .tif file so it is viewed from different angle (example: xz -> xy )

