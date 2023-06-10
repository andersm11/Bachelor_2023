import numpy as np
import tifffile

input = "C:\\Users\\ahmm9\\Desktop\\segmentation_things\\spermcell_3d_v2.tif"
output = "C:\\Users\\ahmm9\\desktop\\segmentation_things\\spermcell_3d_v2_flipped"

# Load the 4D image
tif_image = tifffile.imread(input)
print(tif_image.shape)
# Get the shape of the image
x_padding = 16
y_padding = 16
z_padding = 16
image = tif_image[:, y_padding:-y_padding, z_padding:-z_padding , x_padding:-x_padding]

# Update the dimensions after removing the padding
timesteps, z_axis, x_axis, _ = image.shape

# Reshape the image to swap the z and y axes
reshaped_image = np.reshape(image, (timesteps, z_axis, x_axis, -1))

# Transpose the image to swap the z and y axes
transposed_image = np.transpose(reshaped_image, (0, 2, 1, 3))

# Save the transposed image
tifffile.imwrite(output + ".tif", transposed_image)
