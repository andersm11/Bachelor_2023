import numpy as np
from sklearn.decomposition import PCA
import math
import patch_manager as pmang
from skimage import io
from PIL import Image
import numpy as np


sperm_path  = 'sperm_0p2_70hz_6t_00064.BTF'   # Dataset 2 - 400x400x6  ()  [209,3]
save_path  = ''                         # Dataset 2

sperm_path2 = 'sperm_0p2_40hz_12t_00068.BTF'  # Dataset 1 - 348x348x12 -> 348  / 141 * 20 / 12 = 4.11 [157,7]
save_path2 = ''                         # Dataset 1

print("loading", sperm_path2)


test = io.imread('segmentation.tif')


Y = io.imread(sperm_path2)

timestep = 50 
layer = 6


def remove_nth_row(X, nth):
    I = len(X)
    J = len(X[0])
    print(I,",",J)

    out = np.copy(X)
    for i in range(I):
        for j in range(J):
            if(not(i % nth == 0)):
                out[i, j] = 0
            else:
                out[i, j] = X[i, j]
    return out

images1 = Y[timestep]


full_path = save_path + str(timestep) + ".npy"
full_path2 = save_path2 + str(timestep) + ".npy"

def create_X_Z(X, time_start, timesteps, start_x, end_x, img_len, z_height=48, nth_row=4, row_in_z_start=1):
    Z = np.zeros((timesteps, (end_x - start_x),z_height, img_len))
    for t in range(timesteps):
        # q is row in image and layer in Z
        for q in range(start_x,end_x):
            row_in_z = row_in_z_start
            #Iterate each layer for a timestep
            for image in X[time_start + t]:
                # j is column
                for j in range(len(image[2])):
                    Z[t][q-start_1][row_in_z][j] = image[q][j]     
                row_in_z = (row_in_z + nth_row) % 48
    return Z

def create_Y_Z(X, time_start, timesteps, start_y, end_y, img_len, z_height=48, nth_row=4, row_in_z_start=1):
    Z = np.zeros((timesteps, img_len,z_height, (end_y - start_y)))
    print(np.shape(Z))

    for t in range(timesteps):
        # q is column in image and layer in Z
        for q in range(start_y,end_y):
            row_in_z = row_in_z_start
            #Iterate each layer for a timestep
            for image in X[time_start + t]:
                # j is column
                for j in range(len(image[2])):
                    Z[t][j][row_in_z][q-start_1] = image[j][q]     
                row_in_z = (row_in_z + nth_row) % 48
    return Z

start_1, end_1 = 150, 285
timesteps = 10


time_start = timestep 
time_end = time_start + timesteps


Z1  = create_X_Z(Y, time_start,timesteps, start_1, end_1, 348, 48, 4, 1)

Z1 = Z1[:,:,:,55:100]

print(np.shape(Z1))
np.save("YZ.npy", Z1)
