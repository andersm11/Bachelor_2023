import os
from skimage import io
from tifffile import imwrite
from PIL import Image
import numpy as np
import patch_manager as pmang
import cv2 as cv

import random
import os
from skimage import io

def save_images_as_3d_tif(folder_path, output_file):
    images = []
    for file in os.listdir(folder_path):
        if file.endswith('.png'):
            image = io.imread(os.path.join(folder_path, file), as_gray=True)
            images.append(image)
    stack = np.stack(images)
    ret,thresh1 = cv.threshold(stack*255,1,255,cv.THRESH_BINARY)

    imwrite(output_file, thresh1.astype(np.uint8), imagej=True)

def save_all_folders_as_3d_tif(root_folder, output_folder):
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            output_file = os.path.join(output_folder, f'{folder_name}.tif')
            save_images_as_3d_tif(folder_path, output_file)


def get_tifs_in_folder(folder_path):

    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tif')]

    # Sort the image files by their file name (assuming the file names are in the format `number.png`)
    image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    images = [io.imread(image_file) for image_file in image_files]
    stack = np.stack(images)
    return stack

def get_tifs_in_folder_g(folder_path):
    images = []
    for file in os.listdir(folder_path):
        if file.endswith('.tif'):
            image = io.imread(os.path.join(folder_path, file))
            images.append(image)
    stack = np.stack(images)
    return stack*255


save_all_folders_as_3d_tif("C:\\Users\\Blomst\\groundtruth\\gt_full","C:\\Users\\Blomst\\groundtruth\\gt_full")
save_all_folders_as_3d_tif("C:\\Users\\Blomst\\groundtruth\\gt_head","C:\\Users\\Blomst\\groundtruth\\gt_head")


gt_full = get_tifs_in_folder("C:\\Users\\Blomst\\groundtruth\\gt_full")*255
labkit_full = get_tifs_in_folder("C:\\Users\\Blomst\\Downloads\\segmentation_things\\seg2\\body")*255
#"C:\\Users\\Blomst\\downloads\\segmentation_things\\body1.tif"
gt_head = get_tifs_in_folder("C:\\Users\\Blomst\\groundtruth\\gt_head")*255

"C:\\Users\\Blomst\\downloads\\segmentation_things\\body1.tif"

labkit_head = get_tifs_in_folder("C:\\Users\\Blomst\\Downloads\\segmentation_things\\seg2\\head")
labkit_head_2 = io.imread("C:\\Users\\Blomst\\downloads\\segmentation_things\\head1.tif")
labkit_body_2 = io.imread("C:\\Users\\Blomst\\downloads\\segmentation_things\\body1.tif")

def get_labkit_head_n(i):
    return labkit_head_2[i*48:(i+1)*48]


def get_labkit_body_n(i):
    return labkit_body_2[i*48:(i+1)*48]

print(gt_full.dtype)
print(labkit_full.dtype)

print("labkit")
print(np.shape(labkit_full))
print(np.shape(labkit_head))
print("groundtruth")
print(np.shape(gt_full))
print(np.shape(gt_head))



labkit_full = np.rot90(labkit_full, 1, (1,2))
labkit_full = np.flip(labkit_full, axis=1)

labkit_head = np.rot90(labkit_head, 1, (1,2))
labkit_head = np.flip(labkit_head, axis=1)

print("groundtruth")
print(np.shape(gt_full))
print(np.shape(gt_head))
#print(np.shape(labkit_full))
#count = 0
#pmang.show_patch_as_image(gt_full[0][8], "123")
#for img in labkit_full[0]:
#    if(count < 25):
#        count = count + 1
#        continue
#    print(count)
#    count = count + 1
#    pmang.show_patch_as_image(img*255, "123")

def save_images_as_3d_tiff(folder_path: str, output_file: str):
    """
    This function takes a folder of images as input, orders them by their file name (assuming the file names are in the format `number.png`), and saves them in a 3D image format that ImageJ can read.
    :param folder_path: str - path to the folder containing the images
    :param output_file: str - path to the output file
    """
    # Get a list of all the image files in the folder
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png') or f.endswith('.jpg')]

    # Sort the image files by their file name (assuming the file names are in the format `number.png`)
    image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    #print(image_files)

    # Read all the images into a list
    images = [io.imread(image_file, as_gray=True) for image_file in image_files]

    first_image = io.imread(image_files[0], as_gray=True)

    black_image_array = np.zeros((44, len(first_image[0])), dtype=np.uint16)

    print(np.shape(images))
    print(np.shape(first_image[35:79,:]))
    print(np.shape(black_image_array))
    print(np.shape(black_image_array))


    new_images = []
    for image in images:

        removed = pmang.set_rows_zero(image, 4, 347, random=False)*255
        #pmang.show_patch_as_image(removed*255)
        new_images.append(removed[0:100,:])

    print(np.shape(new_images))
    print("hello")



    # Stack the images into a 3D array
    stack = np.stack(new_images, axis=0).astype(np.uint16)
    print(np.shape(stack))
    #out  = np.ndarray.tolist(stack)

    #print(type(stack))
    #print(type(out))
    # Save the 3D array as a TIFF file
    np.save("C:\\Users\\Blomst\\Documents\\synthetic_Z_karla_original.npy",stack)
    imwrite(output_file, stack, imagej=True)


#sample = io.imread('C:\\Users\\Blomst\\Desktop\\andersv3.tif',as_gray=True)
#print(np.shape(sample))
#

#save_images_as_3d_tiff('C:\\Users\\Blomst\\Documents\\groundtruth', 'C:\\Users\\Blomst\\Documents\\synthetic_Z_karla_original.tiff')

#test = io.imread('C:\\Users\\Blomst\\Documents\\synthetic_Z_karla_original.tiff')
#print(np.shape(test))
#for pix in test:
#    pmang.show_patch_as_image(pix)


# old
#karla = io.imread('C:\\Users\\Blomst\\Documents\\karla_head_gt.tif')
#segme = io.imread('C:\\Users\\Blomst\\Desktop\\head_seg.tif')
x = 00
y = 00

#gt_full = gt_full[:,:,y:y+140,x:(x+280)]
gt_head = gt_head[:,:,y:y+140,x:(x+280)]

#karla_head_side



#pmang.show_patch_as_image(gt_full[0][8]*255, "true")
#pmang.show_patch_as_image(gt_head[0][8]*255, "pred")

#x_biases = range(15,17)
#y_biases = range(15,17)
#z_biases = range(-30,-4)
x_biases = [0]
y_biases = [0]
z_biases = [0]

best_x = 0
best_y = 0
best_z = 0

best_precission = 0

for x_bias in x_biases:
    for y_bias in y_biases:
        for z_bias in z_biases:
            TN = 0
            TP = 0
            FN = 0
            FP = 0
            for n in range(10):
                karla = gt_full[n]
                segme = get_labkit_body_n(n)
                for i in range(0,len(karla)):
                    #if((i-1)%4==0):
                        #z_bias = -4*4-3

                        #y_bias = -4

                        #x_bias = -6

                        true = karla[i]
                        rand = random.randint(0,1)
                        pred = rand
                        #pmang.show_patch_as_image(true*255, "true")
                        #pmang.show_patch_as_image(pred, "pred")
                        
                        for k in range(len(true)):
                            for j in range(len(true[0])):
                                if(true[k+x_bias][j+y_bias]):
                                    if(pred):
                                        TP = TP + 1
                                    else:
                                        FN = FN + 1 
                                else:
                                    if(pred):
                                        FP = FP + 1
                                    else:
                                        TN = TN + 1 
            Precision = TP / (TP + FN)
            Recall = TP / (TP + FP)
            if(Precision > best_precission):
                best_precission = Precision
                best_x = x_bias
                best_y = y_bias
                best_z = z_bias
                print("best: (",best_x,",",best_y,",",best_z,")" )
                print("TN", TN)
                print("TP", TP)
                print("FN", FN)
                print("FP", FP)
                print("Recall", Recall)
                print("Precision", Precision)
                img = np.append(true,pred)
                    
                    #pmang.show_patch_as_image(img*255, "pred")



print(best_x)
print(best_y)
print(best_z)
print(best_precission)