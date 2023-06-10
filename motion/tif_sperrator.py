import os
import tifffile as tiff
import numpy as np


# Path to the input hyperstack TIFF file
input_file = "synthetic_z_348_front_recon_flipped.tif"

# Output directory to save the split TIFF files
output_directory = "segmentation_things\\"


# Number of time steps and slices per time step
num_time_steps = 10
slices_per_time_step = 48

# Number of pixels to crop from the y-axis
crop_x = 0
crop_y = 0
tif_image = tiff.imread(input_file)


# Open the input TIFF file
with tiff.TiffFile(input_file) as tif:
    # Read the TIFF stack
    stack = tif.asarray()
    
    # Check if the stack has the expected dimensions
    
    if len(stack.shape) == 4:
        timesteps, z_axis, x_axis, y_axis = stack.shape
        # Split the 4D image into individual 3D timesteps
        for timestep in range(timesteps):
            # Extract the 3D image for the current timestep
            timestep_image = stack[timestep, :, :, :]

            # Save the 3D image as a separate .tif file
            output_file = os.path.join(output_directory, f"time_step_{timestep+1}.tif")
            tiff.imwrite(output_file, timestep_image)
        exit(1)
    elif len(stack.shape) != 3:
        print("Invalid hyperstack dimensions!")
        exit(1)
    

    num_slices, width, height = stack.shape

    # Calculate the number of slices per time step
    slices_per_time_step = min(slices_per_time_step, num_slices)

    # Crop the stack by removing the specified number of pixels from the y-axis
    cropped_stack = stack[:, :, crop_x:]

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Split the hyperstack into time steps
    for t in range(num_time_steps):
        # Calculate the start and end slice indices for the current time step
        start_slice = t * slices_per_time_step
        end_slice = (t + 1) * slices_per_time_step

        # Extract the slices for the current time step
        time_step_slices = cropped_stack[start_slice:end_slice]

        # Reshape the time step slices to the desired dimensions
        reshaped_slices = np.reshape(time_step_slices, (slices_per_time_step, width, height - crop_x))

        # Create a new TIFF file for the current time step
        output_file = os.path.join(output_directory, f"time_step_{t+1}.tif")

        # Save the time step slices to the output TIFF file
        tiff.imwrite(output_file, reshaped_slices)

    print("Splitting complete!")
