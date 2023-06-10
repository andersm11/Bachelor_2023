import numpy as np
import os
from skimage.filters import threshold_triangle
from skimage.morphology import disk
from skimage.morphology import dilation
from scipy import ndimage
import random
import math
from PIL import Image

def filter_isolated_cells(array, struct):
        """ Return array with completely isolated single cells removed
        :param array: Array with completely isolated single cells
        :param struct: Structure array for generating unique regions
        :return: Array with minimum region size > 1
        """

        filtered_array = np.copy(array)
        id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
        id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1)))
        area_mask = (id_sizes <= 8)
        filtered_array[area_mask[id_regions]] = 0
        return filtered_array


def threshhold_and_dialate(img):
        '''Dialates the given binary image'''
        thresh = threshold_triangle(img)
        binary = img > thresh
        removed_loners = filter_isolated_cells(binary, struct=np.ones((3,3)))
        return dilation(removed_loners, disk(15))
    
    
def binary_to_coordinate(binary, window_size, time, layer):
    '''Convert binary image to coordinates'''
    coordinates = []

    X, Y = binary.shape
    half = int(window_size/2)
    for x in range(half, X-half):
        for y in range(half, Y-half):
            if(binary[x][y]):
                coordinates.append((x- half,y-half, time, layer))
    return coordinates

def random_coordinates(img, n, window_size, time, layer):
    '''Gets random coordinates in image'''
    dialation = threshhold_and_dialate(img[layer])
    img = Image.fromarray(dialation)
    img = img.convert('RGB')
    img.save("binary.png")
    rs = binary_to_coordinate(dialation, window_size, time, layer)
    random.shuffle(rs)
    if(len(rs)<n):
        return rs
    return rs[:n]

def random_coordinates_all(data, n, window_size, max):
    '''Gets random coordinates in 3D image'''
    rs_full= []
    time_max = len(data) - 1
    layer_max= len(data[0]) - 1
    while(len(rs_full) < max):
        time = random.randint(0, time_max)
        layer = random.randint(0, layer_max)
        rs = random_coordinates(data[time], n, window_size, time, layer)
        length = len(rs)
        #print("Time:",time, "layer:",layer, "found", length, "patches")
        for r in rs:
            rs_full.append(r)
    return rs_full

def random_coordinates_all2(data, n, window_size, max):
    '''Gets random coordinates in 3D image'''
    counter = 0
    results = []
    time_min = 0
    time_max = len(data) - 1
    layer_max= len(data[0]) - 1
    while(counter < max):
        time = random.randint(time_min, time_max)
        layer = random.randint(0, layer_max)
        rs = random_coordinates(data[time], n, window_size, time, layer)
        length = len(rs)
        if(length == 99856 or length == 135424): length = 0
        #print("Time:",time, "layer:",layer, "found", length, "patches")
        counter = counter + 1
        results.append((time, layer, length))
    return results



def get_patch(data, x, y, window_size, time=0, layer=0):
        ''' Gets a patch from image'''
        if(layer==0):       image = data
        elif(time==0):      image = data[layer]
        else:               image = data[time][layer]
        patch = image[x:x + window_size, y:y + window_size]
        row = patch.reshape(-1)
        return row

    
    
def get_patch_center(data, x, y, window_size):
        ''' Gets a patch from image'''
        patch = data[x - math.floor(window_size/2):x + math.floor(window_size/2), y - math.floor(window_size/2):y + math.floor(window_size/2)]
        row = patch.reshape(-1)
        return row    
    
def get_patches(data, rs, window_size):
    '''Gets multiple patches from image '''
    patches = []
    c = 0
    n = len(rs)
    for r in rs:
        c=c+1
        if((c % 1000) == 0):
            print(str(c) + "/" + str(n))
        patch = get_patch(data, r[0], r[1], window_size, r[2], r[3])
        if(len(patch) == window_size*window_size):
            patches.append(patch)
    return patches

def get_patches_layers(data, rs, window_size):
    '''Gets mutiple patches from 3D image'''
    patches = [np.array([])]
    for r in rs:
        time = r[2]
        layer = r[3]
        patch = get_patches(data, (r[0], r[1]), window_size)
        if len(patches) == 0:
            patches = patch
        else:
            patches = np.append(patches, patch, axis=0)
    return patches
    
def vec_2_patch(vec, window_size):
    '''Converts patch vector into a patch matrix'''
    return vec.reshape((window_size, window_size))  

def patch_2_vec(patch):
    
    return patch.reshape(-1)

def remove_row_from_patch(patch, n, window_size, mu):
    removed = np.copy(patch)
    for i in range( window_size):
        #removed[row][i] = mu[i]    # Potentiel idé
        mu_index = (n * window_size) + i
        removed[n][i] = (mu[mu_index])       # vil ignorer de principal components vi ikke kender.
    return removed

def set_row_zero(patch, n, window_size):
    removed = np.copy(patch)
    for i in range( window_size):
        #removed[row][i] = mu[i]    # Potentiel idé
        index = i
        removed[n][index] = 0
    return removed

def set_rows_zero(patch, nth, window_size, random=False):
    removed = np.copy(patch)
    if(random):
        randomness = random.randint(0, window_size)
    else:
        randomness = 0
    for n in range(window_size):
        if(not(n % nth == 0)):
            random_n = (n + randomness) % window_size
            removed = set_row_zero(removed,random_n, window_size)
    return removed

def remove_column_from_patch(patch, n, window_size, mu):
    removed = np.copy(patch)
    for i in range( window_size):
        mu_index = (i * window_size) + n
        randomess = [-1,1][random.randrange(2)]
        removed[i][n] = (mu[mu_index]) 
    return removed

def save_patch_as_image(patch, path):
    '''Saves patch as image (.png)'''
    img = Image.fromarray(patch)
    img = img.convert('RGB')
    img.save(path)

def show_patch_as_image(patch, title):
    '''Saves patch as image (.png)'''
    img = Image.fromarray(patch)
    img = img.convert('RGB')
    img.show(title)

def make_directory(path):
    '''Makes new directory at given path if such directory does not already exist'''
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
    return 1

def find_min_and_max(data):
    vector = data.reshape(-1)
    min=255
    max=0
    for elem in vector:
        if(min > elem): min = elem
        if(max < elem): max = elem
    return (min, max)

def change_range(vector, min, max, new_min, new_max):
    length = len(vector) 
    #print(length)
    for i in range(length):
        new_number = (vector[i]-min)/(max - min ) * new_max
        if(new_number>255): new_number = 255.0
        if(new_number<0):   new_number = 0.0
        vector[i] = new_number
    return vector

def change_range_patches(patches, min, max, new_min, new_max):
    new = patches
    length = len(patches)
    for i in range(length):
        if(i%10 == 0): print(i, "/", length)
        new[i] = change_range(new[i], min, max, new_min, new_max)
    print(len(new))
    print(len(new[0]))
    print(new[0])
    return new
