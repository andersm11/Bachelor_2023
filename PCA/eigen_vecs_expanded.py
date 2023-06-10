from reconstructor import Reconstructor
import patch_manager as pmang
from skimage import io
from PIL import Image
import numpy as np

def estim_components(window_size): 
    return 30
    #return int(((6.8995 * window_size) - 56.0418))


X = io.imread('C:\\Users\\ahmm9\\Desktop\\sperm_0p2_70hz_6t_00064.BTF')
img = X[200]
patches_n = 2000
layer = 3
result_folder = "C:\\Users\\ahmm9\\Desktop\\result\\"

rs = pmang.random_coordinates_layers(img,patches_n,32)
for window_s in range(10,33,1):
    
    recon = Reconstructor()
    n_component = estim_components(window_s)
    print("Experiment: " + str(n_component) + ", " + str(window_s))
    
    patches = pmang.get_patches_layers(X[layer],rs,window_s)
    recon.get_pca(patches,window_s)
    
    path1 = result_folder+"eigen_vectors_img\\window_size_"+str(window_s)
    pmang.make_directory(path1)
    klat = pmang.vec_2_patch(pmang.get_patch(img[layer], 210, 108, window_s), window_s)
    #k = 30
    
    k_array = recon.learn_const(klat)
    print(len(k_array))
    for i in range(recon.n_var):
        path = path1+"\\eigenvectors_"+str(i+1)+".png"
        patch = pmang.vec_2_patch(recon.mu + (k_array[i]*np.sqrt(recon.Eig[i])*recon.Vecs[i]),window_s)
        pmang.save_patch_as_image(patch,path)