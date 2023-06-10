from reconstructor import Reconstructor
import patch_manager as pmang
from skimage import io
from PIL import Image
import numpy as np
#import histogram as his
import math

X = io.imread('C:\\Users\\ahmm9\\Desktop\\sperm_0p2_70hz_6t_00064.BTF')
X2 = X[200]
img= X2[3]
img = X2[3]
result_folder = "C:\\Users\\ahmm9\\Desktop\\result\\"
layer = 3
time_step = 6
patches_n = 2000
WINDOW_SIZE = 32
FULL_IMAGE_SIZE = len(img)
pmang.make_directory(result_folder+"og_patch")
pmang.save_patch_as_image(img,result_folder+"og_patch\\org.png")

his.make_histogram(img)

for wz in (range(2,32,2)):

    rs = pmang.random_coordinates_all(X,patches_n,wz,5000)

    recon = Reconstructor()

    patches = pmang.get_patches(X, rs, wz)
    recon.get_pca(patches,wz)

    print("Experiment: " + str(recon.n_var) + ", " + str(wz))
    mu_path = result_folder + "mu"
    pmang.make_directory(mu_path)
    pmang.save_patch_as_image(pmang.vec_2_patch(recon.mu,wz),mu_path+"\\"+str(1)+"mu.png")

    img_vec = img.reshape(-1)
    img_output = img
    for i in range(len(img)):
        for j in range(len(img[0])):
            x = i
            y = j
            if (x + wz > FULL_IMAGE_SIZE):
                over_spill = ((x+wz)-FULL_IMAGE_SIZE)
                x = x - over_spill
            if (y + wz > FULL_IMAGE_SIZE):
                over_spill = ((y+wz)-FULL_IMAGE_SIZE)
                y = y - over_spill
            patch = pmang.get_patch(img,x,y,wz)
            rec_patch = recon.reconstruct_patch(patch,wz)
            rec_patch = pmang.vec_2_patch(rec_patch,wz)
            pmang.make_directory(result_folder + "reconstructed_patches")
            pmang.save_patch_as_image(rec_patch, result_folder + "reconstructed_patches\\patch_" + str(i) + "_" + str(wz) + ".png")
            patch_avg = np.average(rec_patch)
            #print(patch_avg)
            img_output[i][j] = patch_avg
        print(i)

    pmang.make_directory(result_folder+"full_image")
    pmang.save_patch_as_image(img_output,result_folder+"full_image\\full_recon" + str(wz) + ".png")
    his.make_histogram(img_output)