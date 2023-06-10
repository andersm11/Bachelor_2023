from reconstructor import Reconstructor
import patch_manager as pmang
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt 
import patch_manager as pmang
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def make_histogram(patch):
    v_patch = patch.reshape(-1)
    avg = np.mean(v_patch)
    avg_vec = np.full(v_patch.shape,avg)
    plt.figure()

    
    plt.subplot(1,2,1)
    plt.title("Pixel color scale")
    plt.xlabel("pixul number")
    plt.ylabel("Grayscale value")
    plt.plot(v_patch)
    plt.plot(avg_vec,color='r')
    v_patch = np.flip(v_patch)
    plt.subplot(1,2,2)
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.xlim([0.0,255])
    histogram, bin_edges = np.histogram(v_patch, bins=256,range=(0, 255))
    plt.plot(bin_edges[0:-1], histogram) 
    histogram, bin_edges = np.histogram(avg_vec,bins=256,range=(0, 255))
    plt.show()
    

X = io.imread('C:\\Users\\ahmm9\\Desktop\\sperm_0p2_70hz_6t_00064.BTF')
X = X[0:10]
counter = np.zeros(512, np.uint16)


def distribution_counter(X):
    vector = X.reshape(-1)
    length = len(vector)/10000
    counter_n = 0
    for elem in vector:
        if(counter_n%10000 == 0):
            print(counter_n, "/", length)
            counter_n = counter_n + 1
        counter[elem] = counter[elem] + 1



distribution_counter(X)
for i in range(len(counter)):
    print(i, ",", counter[i])

#make_histogram(X)