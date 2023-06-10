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

print("loading", sperm_path)
X = io.imread(sperm_path)
Y = io.imread(sperm_path2)

#timestep = 157 #Time: 157 layer: 7 found 10000 patches YYYYY

timestep = 797 #Time: 209 layer: 3 found 10000 patches
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
images2 = X[timestep]


#(timestep, layers, y, x)

#pmang.show_patch_as_image(X[timestep][layer])
#experiment = remove_nth_row(X[timestep][layer], 7)
#pmang.show_patch_as_image(experiment)
#np.save("C:\\Users\\Blomst\\z\\synthetic_400x400x6.npy", experiment)


full_path = save_path + str(timestep) + ".npy"
full_path2 = save_path2 + str(timestep) + ".npy"


#np.save(full_path2, Y[timestep][7])
print("yeet")




start_1, end_1 = 0, 249

timesteps = 3


time_start = timestep 
time_end = time_start + timesteps


Z1 = np.zeros((timesteps, (end_1 - start_1),48, 400))
#Z2 = np.zeros((timesteps, 2,50, 400))

for t in range(timesteps):
    for q in range(start_1,end_1):
        c1 = 5
        for image in X[time_start + t]:
            for j in range(len(image[2])):
                Z1[t][q-start_1][c1][j] = image[q][j]     #[104-114][alle][135-165][alle]
            c1 = (c1 + 7) % 50


np.save(full_path, Z1)
print(np.shape(Z1))


pmang.show_patch_as_image(Z1[0,0])

exit()


print("done")

c2 = 4
for image in images2:
    for j in range(len(image[2])):
        Z2[c2][j] = image[144][j]
    c2 = (c2 + 7) % 50

#np.save(full_path2, Z2)
