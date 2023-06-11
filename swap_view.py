import numpy as np
import tifffile

input = "spermcell_3d_v2.tif"
output = "spermcell_3d_v2_flipped"


tif_image = tifffile.imread(input)
print(tif_image.shape)


x_padding = 16
y_padding = 16
z_padding = 16

image = tif_image[:, y_padding:-y_padding, z_padding:-z_padding , x_padding:-x_padding]


timesteps, z_axis, x_axis, _ = image.shape


reshaped_image = np.reshape(image, (timesteps, z_axis, x_axis, -1))


transposed_image = np.transpose(reshaped_image, (0, 2, 1, 3))

# Save the transposed image
tifffile.imwrite(output + ".tif", transposed_image)
