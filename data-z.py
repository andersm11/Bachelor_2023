import numpy as np
from sklearn.decomposition import PCA
import math
import patch_manager as pmang
from skimage import io
from PIL import Image
import numpy as np


sperm_path  = 'C:\\Users\\Blomst\\sperm_0p2_70hz_6t_00064.BTF'   # Dataset 1 - 400x400x6  ()  [209,3]
save_path  = 'C:\\Users\\Blomst\\z\\1\\'                         # Dataset 1

sperm_path2 = 'C:\\Users\\Blomst\\sperm_0p2_40hz_12t_00068.BTF'  # Dataset 0 - 348x348x12 -> 348  / 141 * 20 / 12 = 4.11 [157,7]
save_path2 = 'C:\\Users\\Blomst\\z\\0\\'                         # Dataset 0

print("loading", sperm_path2)
#X = io.imread(sperm_path)

test = io.imread('c:\\users\\blomst\\documents\\segmentationbaby.tif')
print(np.shape(test))

Y = io.imread(sperm_path2)

timestep = 50 #Time: 209 layer: 3 found 10000 patches
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
#images2 = X[timestep]


#(timestep, layers, y, x)

#pmang.show_patch_as_image(X[timestep][layer])
#experiment = remove_nth_row(X[timestep][layer], 7)
#pmang.show_patch_as_image(experiment)
#np.save("C:\\Users\\Blomst\\z\\synthetic_400x400x6.npy", experiment)


full_path = save_path + str(timestep) + ".npy"
full_path2 = save_path2 + str(timestep) + ".npy"


#np.save(full_path2, Y[timestep][7])
print("yeet")


def create_X_Z(X, time_start, timesteps, start_x, end_x, img_len, z_height=48, nth_row=4, row_in_z_start=1):
    # Z = (time, x, z, 400)
    Z = np.zeros((timesteps, (end_x - start_x),z_height, img_len))
    for t in range(timesteps):
        # q is row in image and layer in Z
        for q in range(start_x,end_x):
            row_in_z = row_in_z_start
            #Iterate each layer for a timestep
            for image in X[time_start + t]:
                # j is column
                for j in range(len(image[2])):
                    Z[t][q-start_1][row_in_z][j] = image[q][j]     #[104-114][alle][135-165][alle]
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
                    Z[t][j][row_in_z][q-start_1] = image[j][q]     #[104-114][alle][135-165][alle]
                row_in_z = (row_in_z + nth_row) % 48
    return Z


#rows
start_1, end_1 = 150, 285
timesteps = 10


time_start = timestep 
time_end = time_start + timesteps


Z1  = create_X_Z(Y, time_start,timesteps, start_1, end_1, 348, 48, 4, 1)

Z1 = Z1[:,:,:,55:100]

print(np.shape(Z1))
np.save("c:\\users\\blomst\\YZ.npy", Z1)

exit()
Z1 = np.zeros((timesteps, (end_1 - start_1),48, 348))
#Z2 = np.zeros((timesteps, 2,50, 400))

for t in range(timesteps):
    # q is rows
    for q in range(start_1,end_1):
        c1 = 1
        #Iterate each layer for a timestep
        for image in Y[time_start + t]:
            # j is column
            for j in range(len(image[2])):
                Z1[t][q-start_1][c1][j] = image[q][j]     #[104-114][alle][135-165][alle]
            c1 = (c1 + 4) % 48


#create_Y_Z(Y, time_start,timesteps, start_1, end_1, 348, 48, 4, 1)



pmang.show_patch_as_image(Z1[0,0])

exit()


print("done")

c2 = 4
for image in images2:
    for j in range(len(image[2])):
        Z2[c2][j] = image[144][j]
    c2 = (c2 + 7) % 50

#np.save(full_path2, Z2)
