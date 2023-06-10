import numpy as np
from scipy.ndimage import binary_dilation
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tifffile import imwrite, imread
from skimage import io
from skimage.morphology import skeletonize_3d


def tiff_to_binary_array(file_path):
    # Load the 3D TIFF image
    tiff_image = io.imread(file_path)
    
    # Convert the image to a binary array
    binary_array = np.where(tiff_image > 0, 1, 0)
    
    return binary_array

#visualize skeleton 
def visualize_3d_skeleton(skeleton, head, point):
    x1, y1, z1 = skeleton
    x2, y2, z2 = head
    x,y,z = point
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x1, y1, z1, alpha=0.1) #skeleton
    ax.scatter3D(x2, y2, z2, alpha=0.05) #head
    ax.plot(0,0,0,'bo')
    ax.plot(x,y,z,'bo')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal', adjustable = 'datalim')
    plt.show()


def find_ones_3d(array):
    ones = []
    for i in range(len(array)):
        for j in range(len(array[i])):
            for k in range(len(array[i][j])):
                if array[i][j][k] == 1:
                    ones.append((i, j, k))
    return ones

def find_ones_2d(array):
    ones = []
    for i in range(len(array)):
        for j in range(len(array[i])):
                if array[i][j] == 1:
                    ones.append((i, j))
    return ones

def load_grayscale_image_as_binary(path):
    image = imread(path)
    gray_image = np.mean(image, axis=2)
    return gray_image > 128

def dilate_image_2D(binary_array, iter):
    binary_array = np.array(binary_array)
    dilated_array = binary_dilation(binary_array, iterations=iter)
    return dilated_array


save_tail_points = input("Save files? (y/n): ")

#Load head and full segmentation
# range of t indicates how many timesteps there is.
for t in range(1,11):
    head_folder_path = "path"
    full_folder_path = "path"
    head1 = head_folder_path+"\\time_step_"+str(t)+".tif"
    full1 = full_folder_path+"\\time_step_"+str(t)+".tif"

    #Read files and turn into binary array
    binary_head = tiff_to_binary_array(head1) 
    binary_full = tiff_to_binary_array(full1)


    # Projects the full segmentation onto the binary heads principalcomponents
    # The head is centered at (0,0), and the first principal component is aligned along the x-axis
    def PCA_project_3D(binary_head, binary_full, is_head = False):
        binary_full_coordinates = find_ones_3d(binary_full)
        X = find_ones_3d(binary_head)
        pca = PCA(n_components=3)
        pca.fit(X)
        data_centered = binary_full_coordinates-pca.mean_
        x = np.dot(data_centered, pca.components_[0])
        y = np.dot(data_centered, pca.components_[1])
        z = np.dot(data_centered, pca.components_[2])
        return x, y ,z

    skelleton = skeletonize_3d(binary_full)
    x1, y1, z1 = PCA_project_3D(binary_head, skelleton)
    x2, y2, z2 = PCA_project_3D(binary_head, binary_head,is_head=True)



    placement = -40

    best = 0
    distance = 1000
    #Finds the intersection 
    if np.mean(x1) > 0 :
        placement = 40
    for i in range(len(x1)):
        dif = abs(x1[i] - placement)
        if( dif < distance):
            distance = dif
            best = i
        
    #Saves or displays output
    output_path = "path"
    if save_tail_points == "yes" or save_tail_points == "y":
        np.save(output_path+"\\point"+str(t),np.array(([x1[best], y1[best], z1[best]])))
    else:
        visualize_3d_skeleton((x1, y1, z1), (x2, y2, z2), (x1[best], y1[best], z1[best]))



