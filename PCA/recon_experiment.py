from reconstructor import Reconstructor
import patch_manager as pmang
from skimage import io
from PIL import Image
import numpy as np
from sklearn.metrics import mean_squared_error


def experiment (img, name, recon: Reconstructor):
        recon_arr = recon.reconstruct_patch(img, 150)
        patch = pmang.vec_2_patch(recon_arr, window_size)
        org   = pmang.patch_2_vec(img)


        img = Image.fromarray(patch)
        #path = "C:\\Users\\Loh\\Desktop\\Bachelor\\pca_whitenoise_images_layer_" + str(n) + "_k_" + str(k) + "_window_" + str(window_size) +"\\"
        path = result_folder +"reconstruction_" + name  + "\\"
        pmang.make_directory(path)
        path = path + str(recon.n_var)+ "_" + str(window_size) + "x" + str(window_size) +  ".png"
        print(path)
        img = img.convert('RGB')
        print("SAve")
        img.save(path)
        
        
        
n_components = [75,  91, 107, 127, 100, 135, 133, 176, 204, 225, 272, 224, 282, 300, 418, 333, 406, 399, 420, 495, 418, 446, 579]
window_sizes = range(10, 33, 1)


sperm_path ='C:\\Users\\ahmm9\\Desktop\\sperm_0p2_70hz_6t_00064.BTF'
print("loading", sperm_path)
X = io.imread(sperm_path)
result_folder = "C:\\Users\\ahmm9\\Desktop\\result\\"
layer = 3
time_step = 200
patches_n = 500
window_size = 32
        
#pmang.make_directory(result_folder+"og_patch")
#pmang.save_patch_as_image(X[layer],result_folder+"og_patch\\org.png")

rs = pmang.random_coordinates_all(X,patches_n,window_size, max=2000)
#print(len(rs))
#print(rs[0])

for i in range(len(n_components)):
        recon = Reconstructor()
        
        if(i != 22):
                continue

        n_component = n_components[i]
        #n_component = n_components[i]
        window_size = window_sizes[i]
        
        patches = pmang.get_patches(X, rs, window_size)
        print(len(patches))
        print(len(patches[0]))

        min, max = 70, 170
        print(min)
        print(max)
        print("Changing range")
        patches = pmang.change_range_patches(patches, min, max, 0, 255)
        #patches = np.load("C:\\Users\\Blomst\\100k_patches_scaled_70_170_32x32.npy")
        print("Saving")
        #np.save("C:\\Users\\Senio\\10k_patches_scaled_70_170_32x32.npy", patches)
        
        print("PCA")
        recon.get_pca(patches,window_size, variance = 1.0, n_componets=800)
        #recon.save_principal_components_as_image(10, "c:\\Users\\senio\\range\\" ,window_size)
        #pmang.save_patch_as_image(recon.reconstruct_image(X[200][3], 250, window_size), "e:\\full_restoration.png")
        
        print("Experiment: " + str(recon.n_var) + ", " + str(window_size))

        mu_path = result_folder + "mu_range"
        #pmang.make_directory(mu_path)
        #pmang.save_patch_as_image(pmang.vec_2_patch(recon.mu,window_size),mu_path+"\\"+str(i)+"mu.png")


        print(pmang.get_patch(X, 210, 108, window_size, time_step, layer))

        streg = pmang.vec_2_patch(pmang.get_patch(X, 190, 120,  window_size, time_step, layer), window_size)
        klat = pmang.vec_2_patch(pmang.get_patch(X, 210, 108, window_size, time_step, layer), window_size)
        ingen = pmang.vec_2_patch(pmang.get_patch(X, 50, 52,  window_size, time_step, layer), window_size)
        
        min = 70
        max = 170

        changed_range_streg = pmang.change_range_patches(streg, min, max, 0, 255)
        changed_range_klat = pmang.change_range_patches(klat, min, max, 0, 255)
        changed_range_ingen = pmang.change_range_patches(ingen, min, max, 0, 255)

        streg_float = changed_range_streg.astype(np.float64)
        klat_float = changed_range_klat.astype(np.float64)
        ingen_float = changed_range_ingen.astype(np.float64)

        streg_removed = pmang.remove_row_from_patch(streg_float, 0, window_size, recon.mu)
        streg_removed = pmang.remove_row_from_patch(streg_removed, 11, window_size, recon.mu)
        streg_removed = pmang.remove_row_from_patch(streg_removed, 13, window_size, recon.mu)

        klat_removed = pmang.remove_column_from_patch(klat_float, 18, window_size, recon.mu)
        #klat_removed = pmang.remove_row_from_patch(klat_removed, 10, window_size, recon.mu)
        #klat_removed = pmang.remove_row_from_patch(klat_removed, 13, window_size, recon.mu)

        ingen_removed = pmang.remove_row_from_patch(ingen_float, 9, window_size, recon.mu)
        ingen_removed = pmang.remove_row_from_patch(ingen_removed, 11, window_size, recon.mu)
        ingen_removed = pmang.remove_row_from_patch(ingen_removed, 13, window_size, recon.mu)

        #print(streg_removed)

        

        experiment(klat_removed, "klat_removed_range_1",recon)
        #experiment(streg_removed, "streg_removed_3",recon)
        #experiment(ingen_removed, "ingen_removed_3",recon)

        experiment(klat, "klat_range",recon)
        #experiment(streg, "streg",recon)
        #experiment(ingen, "ingen",recon)