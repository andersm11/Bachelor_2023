from reconstructor import Reconstructor
import patch_manager as pmang
from skimage import io
from PIL import Image
import numpy as np
import histogram as his
def experiment (img, name, recon: Reconstructor,offsets,result_img):
        offset_x = offsets[0]
        offset_y = offsets[1]
        recon_arr = recon.reconstruct_patch(img)
        print(len(recon_arr))
        break_off = 0
        break_jump = (FULL_IMAGE_SIZE - WINDOW_SIZE)
        break_offset = 0
        for x in range(len(recon_arr)):
                if break_off == WINDOW_SIZE:
                        break_off = 0
                        break_offset += break_jump        
                result_img[offset_x+offset_y+x+break_offset] = recon_arr[x]
                break_off += 1
        patch = pmang.vec_2_patch(recon_arr, WINDOW_SIZE)



        img = Image.fromarray(patch)
        #path = "C:\\Users\\Loh\\Desktop\\Bachelor\\pca_whitenoise_images_layer_" + str(n) + "_k_" + str(k) + "_window_" + str(window_size) +"\\"
        path = result_folder +"reconstruction_" + name   + "\\"
        pmang.make_directory(path)
        path = path + str(recon.n_var)+ "_" + str(WINDOW_SIZE) + "x" + str(WINDOW_SIZE) + str(offset_x+offset_y)+  ".png"
        img = img.convert('RGB')
        img.save(path)
        
        
        
        
n_components = [75,  91, 107, 127, 100, 135, 133, 176, 204, 225, 272, 224, 282, 300, 418, 333, 406, 399, 420, 495, 418, 446, 579]
window_sizes = range(10, 33, 1)

X = io.imread('C:\\Users\\ahmm9\\Desktop\\sperm_0p2_70hz_6t_00064.BTF')
X = X[200]
lay = X[3]
cropped = lay
result_folder = "C:\\Users\\ahmm9\\Desktop\\result\\"
layer = 3
time_step = 6
patches_n = 5000
WINDOW_SIZE = 16
FULL_IMAGE_SIZE = len(lay)

pmang.make_directory(result_folder+"og_patch")
pmang.save_patch_as_image(cropped,result_folder+"og_patch\\org.png")

crop_copy = cropped.reshape(-1)

rs = pmang.random_coordinates_layers(X,patches_n,WINDOW_SIZE)

recon = Reconstructor()

#n_component = n_components[i]
#window_size = window_sizes[i]

patches = pmang.get_patches_layers(X, rs, WINDOW_SIZE)
recon.get_pca(patches,WINDOW_SIZE)

print("Experiment: " + str(recon.n_var) + ", " + str(WINDOW_SIZE))
mu_path = result_folder + "mu"
pmang.make_directory(mu_path)
pmang.save_patch_as_image(pmang.vec_2_patch(recon.mu,WINDOW_SIZE),mu_path+"\\"+str(1)+"mu.png")
offsetx = 0
offsety = 0





for i in range(0,len(cropped),WINDOW_SIZE):
        for j in range(0,len(cropped[0]),WINDOW_SIZE):
                patch = pmang.vec_2_patch(pmang.get_patch(cropped,i,j,WINDOW_SIZE),WINDOW_SIZE)
                
                experiment(patch, "klat",recon,(offsetx,offsety),crop_copy)
                offsetx += WINDOW_SIZE
        offsetx = 0
        offsety += WINDOW_SIZE * FULL_IMAGE_SIZE

pmang.make_directory(result_folder+"full_image")
full_image = pmang.vec_2_patch(crop_copy,FULL_IMAGE_SIZE)
pmang.save_patch_as_image(full_image,result_folder+"full_image\\full_recon.png")
