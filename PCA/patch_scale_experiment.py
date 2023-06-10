from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from skimage import io
from sklearn.decomposition import PCA
import random
from numpy.linalg import svd
import os
import matplotlib.pyplot as plt 
from skimage.filters import threshold_triangle
from skimage.morphology import disk
from skimage.morphology import dilation
from scipy import ndimage



#Read tiff
X = io.imread('sperm_0p2_70hz_6t_00064.BTF')  # (3000, 12, 400, 400)

X = X[200]


# generate random coordinates
def filter_isolated_cells(array, struct):
    #https://stackoverflow.com/questions/28274091/removing-completely-isolated-cells-from-python-array
    """ Return array with completely isolated single cells removed
    :param array: Array with completely isolated single cells
    :param struct: Structure array for generating unique regions
    :return: Array with minimum region size > 1
    """

    filtered_array = np.copy(array)
    id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
    id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1)))
    area_mask = (id_sizes <= 5)
    filtered_array[area_mask[id_regions]] = 0
    return filtered_array


def threshhold_and_dialate(img):
    thresh = threshold_triangle(img)
    binary = img > thresh
    removed_loners = filter_isolated_cells(binary, struct=np.ones((3,3)))
    return dilation(removed_loners, disk(30))

def random_coordinates(img, n, window_size):
    dialation = threshhold_and_dialate(img)
    img = Image.fromarray(dialation)
    img = img.convert('RGB')
    img.save("Dialated.png")
    rs = binary_to_coordinate(dialation, window_size)
    random.shuffle(rs)
    return rs[:n]

def binary_to_coordinate(binary, window_size):
    coordinates = []

    X, Y = binary.shape
    X = X - window_size
    Y = Y - window_size
    for x in range(X):
        for y in range(Y):
            if(binary[x][y]):
                coordinates.append((x,y))
    return coordinates



# Get a patch from array given windows_size and top-right corner in x,y
def get_patch_jon(data, x, y, window_size):
    patch = data[x:x + window_size, y:y + window_size]
    row = patch.reshape(-1)
    row.shape = (len(row),1)
    return row
        


def get_patch(data, x, y, window_size):
    patch = data[x:x + window_size, y:y + window_size]
    row = patch.reshape(-1)
    return row
        
# Get patches and format into PCA ready array, given coordinate set rs and windows_size
def get_patches(data, rs, window_size):
    patches = np.array([])
    for r in rs:
        patch = get_patch(data, r[0], r[1], window_size)
        if len(patches) == 0:
            patches = np.array([patch])
        else:
            patches = np.append(patches, [patch], axis=0)
    return patches

    
def get_component_count(window_size, rs):
    
    patches = get_patches(X[3], rs, window_size)


    pca = PCA(n_components=window_size*window_size)
    pca.fit(patches)

    var = pca.explained_variance_ratio_[0:50]
    ran = range(1, len(var) + 1)

    sum_var = 0.0
    for i in range(len(pca.singular_values_)):

        if(sum_var > 0.95):
            return i
            break
        var = pca.explained_variance_ratio_[i]
        #print(i ," : " , var)
        sum_var = sum_var + var


def display_component_count():
    result = np.array([])
    n=3
    patches_n = 8192
    dilation = threshhold_and_dialate(X[n])

    for j in range(1):
        comp_count = []
        count = 0   
        for i in range(10,33,2):
            print(str(j) + ", " + str(i))
            count +=2
            rs = random_coordinates(dilation, patches_n, i)
            n = get_component_count(i, rs)
            comp_count = np.append(comp_count,n/i**2)
            
        if(len(result) == 0):
            result = np.array([comp_count])
        else:
            result = np.append(result, [comp_count], axis=0)
        print(result)
    
    result = np.mean(result, axis=0)
    ran = range(10, len(result) + 10)
    plt.plot(ran, result, 'ro')
    plt.xlabel("Patch size")
    plt.ylabel("Components")
    plt.show()
    return result

display_component_count()
