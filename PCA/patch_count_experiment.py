from reconstructor import Reconstructor
import patch_manager as pmang
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt 

X = io.imread('C:\\Users\\ahmm9\\Desktop\\sperm_0p2_70hz_6t_00064.BTF')
X = X[200]

sample_counts = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
window_size = 12
results = []

for count in sample_counts:
    print(count)
    recon = Reconstructor()
    rs = pmang.random_coordinates_layers(X,count,window_size)
    patches = pmang.get_patches_layers(X, rs, window_size)
    recon.get_pca(patches,window_size)
    print(recon.n_var)
    results.append(recon.n_var)
    del recon

plt.plot(results)
plt.xlabel("Sample counts")
plt.ylabel("comps needed for 90 procent")
plt.show()