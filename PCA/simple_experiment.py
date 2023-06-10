from reconstructor import Reconstructor
import patch_manager as pmang
from skimage import io
from PIL import Image
import numpy as np
from sklearn.metrics import mean_squared_error

window_size=3

patch_left = [1,0,0,1,0,0,1,0,0]
patch_midl = [0,1,0,0,1,0,0,1,0]
patch_righ = [0,0,1,0,0,1,0,0,1]
patch_none = [0,0,0,0,0,0,0,0,0]

patch_left_bad = [1,0,0,1,0,0,0,0,0]

patches = [patch_left, 
           patch_midl, 
           patch_righ,
           patch_none]

recon = Reconstructor()

recon.get_pca(patches, window_size, variance=0.8, n_componets=3)
E = recon.Eig
V = recon.Vecs
mu = recon.mu

for i in range(len(V)):
    print(mu + (V[i]*np.sqrt(E[i])))
print()

path = "e:\\simple_eigen_vectors_img\\"
#recon.save_principal_components_as_image(3, path, 3)

good = recon.reconstruct_patch(np.array(patch_left))

good_patch = pmang.vec_2_patch(good, 3)

bad = recon.reconstruct_patch(np.array(pmang.remove_row_from_patch(good_patch, 2, 3, mu)))

bad = bad * 255

bad_patch = pmang.vec_2_patch(bad, 3)


print("Reconstruction of good patch")
print(good_patch)

print("Reconstruction of bad patch")
print(bad_patch)

pmang.save_patch_as_image(bad_patch, path= "e:\\bad.png")