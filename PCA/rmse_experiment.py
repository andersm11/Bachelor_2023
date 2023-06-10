from reconstructor import Reconstructor
import patch_manager as pmang
from skimage import io
from PIL import Image
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def experiment (img, name, recon: Reconstructor, n_components):
        recon_arr = recon.reconstruct_patch(img, n_components=n_components)
        patch = pmang.vec_2_patch(recon_arr, window_size)
        org   = pmang.patch_2_vec(img)

        MSE = mean_squared_error(img[18], patch[18])
        RMSE = np.sqrt(MSE)
        #print("RMSE" , RMSE, "for", name)


        img = Image.fromarray(patch)
        #path = "C:\\Users\\Loh\\Desktop\\Bachelor\\pca_whitenoise_images_layer_" + str(n) + "_k_" + str(k) + "_window_" + str(window_size) +"\\"
        path = result_folder +"reconstruction_" + name  + "\\"
        pmang.make_directory(path)
        path = path + str(n_components)+ "_" + str(window_size) + "x" + str(window_size) +  ".png"
        #print(path)
        img = img.convert('RGB')
        #print("SAve")
        img.save(path)
        return RMSE
        
        
        
n_components = [75,  91, 107, 127, 100, 135, 133, 176, 204, 225, 272, 224, 282, 300, 418, 333, 406, 399, 420, 495, 418, 446, 579]
window_sizes = range(10, 33, 1)

X = io.imread('C:\\Users\\Blomst\\sperm_0p2_70hz_6t_00064.BTF')
result_folder = "E:\\rmse\\"
layer = 3
time_step = 200
patches_n = 200
window_size = 30
        
#pmang.make_directory(result_folder+"og_patch")
#pmang.save_patch_as_image(X[layer],result_folder+"og_patch\\org.png")


rs = pmang.random_coordinates_all(X,patches_n,window_size, max=1000)

recon = Reconstructor()
patches = pmang.get_patches(X, rs, window_size)
recon.get_pca(patches,window_size, variance = 1.0)

result = []

for i in range(10, 900):
        n_component = i
        #recon.save_principal_components_as_image(15, "e:\\random\\" ,window_size)
        
        print("Experiment: " + str(i))

        mu_path = result_folder + "mu"
        pmang.make_directory(mu_path)
        pmang.save_patch_as_image(pmang.vec_2_patch(recon.mu,window_size),mu_path+"\\"+str(i)+"mu.png")

        streg = pmang.vec_2_patch(pmang.get_patch(X, 190, 120,  window_size, time_step, layer), window_size)
        klat = pmang.vec_2_patch(pmang.get_patch(X, 210, 108, window_size, time_step, layer), window_size)
        ingen = pmang.vec_2_patch(pmang.get_patch(X, 50, 52,  window_size, time_step, layer), window_size)

        streg_float = streg.astype(np.float64)
        klat_float = klat.astype(np.float64)
        ingen_float = ingen.astype(np.float64)

        streg_removed = pmang.remove_row_from_patch(streg_float, 0, window_size, recon.mu)
        streg_removed = pmang.remove_row_from_patch(streg_removed, 11, window_size, recon.mu)
        streg_removed = pmang.remove_row_from_patch(streg_removed, 13, window_size, recon.mu)

        klat_removed = pmang.remove_row_from_patch(klat_float, 18, window_size, recon.mu)
        #klat_removed = pmang.remove_row_from_patch(klat_removed, 10, window_size, recon.mu)
        #klat_removed = pmang.remove_row_from_patch(klat_removed, 13, window_size, recon.mu)

        ingen_removed = pmang.remove_row_from_patch(ingen_float, 9, window_size, recon.mu)
        ingen_removed = pmang.remove_row_from_patch(ingen_removed, 11, window_size, recon.mu)
        ingen_removed = pmang.remove_row_from_patch(ingen_removed, 13, window_size, recon.mu)

        #print(streg_removed)

        

        RMSE_removed = experiment(klat_removed, "klat_removed_1", recon, n_component)
        #experiment(streg_removed, "streg_removed_3",recon)
        #experiment(ingen_removed, "ingen_removed_3",recon)

        RMSE_org = experiment(klat, "klat", recon, n_component)
        #experiment(streg, "streg",recon)
        #experiment(ingen, "ingen",recon)
        result.append([i, RMSE_org, RMSE_removed])

#print(result)
np_result = np.array(result)

n = np_result[:,0]
rmse_original = np_result[:,1]
rmse_removed1 = np_result[:,2]
plt.plot(n, rmse_original, 'r--')
plt.plot(n, rmse_removed1, 'b--')
plt.axis([0, 900, 0, 50])
plt.show()