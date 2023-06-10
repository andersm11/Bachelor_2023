import patch_manager as pmang
from skimage import io
from PIL import Image
import numpy as np
from sklearn.metrics import mean_squared_error

sperm_path ='C:\\Users\\Blomst\\sperm_0p2_70hz_6t_00064.BTF'    # Dataset 1 - 400x400x6
sperm_path2 ='C:\\Users\\Blomst\\sperm_0p2_40hz_12t_00068.BTF'  # Dataset 0 - 348x348x12
print("loading", sperm_path)
X = io.imread(sperm_path)
print(np.shape(X))
window_size = 32



#snippet = X[0,0]

def save_last_time(input, output, n):
    print("loading", sperm_path)
    data = io.imread(input)

    data = data[-n:]
    time = 0
    for timestep in data:
        for i in range(len(timestep)):
            layer = timestep[i]
            out  = output + "\\" + str(time) + "\\" + str(i) + ".png"
            dir  = output + "\\" + str(time) + "\\"
            pmang.make_directory(dir)
            pmang.save_patch_as_image(layer, out)
        time = time + 1

#save_last_time(sperm_path2, "c:\\users\\blomst\\seg348x348", 100)
#exit()

def grayscale_2_rgb(patch):
    x,y = patch.shape
    new = np.ndarray((x,y,1))
    for i in range(len(patch)):
        for j in range(len(patch[i])):
            number = patch[i][j]
            new[i][j] = number
    return new


#¤formatted = grayscale_2_rgb(snippet)
#print(np.shape(formatted))


#np.save("recreate.npy", formatted)

#(time, layer, count)
resultater = pmang.random_coordinates_all2(X, 250000, window_size, max=200)

def middle(n):  
    return n[2]    
     
# function to sort the tuple     
def sort(list_of_tuples):  
    return sorted(list_of_tuples, key = middle, reverse=True)  
     
 
sorted = sort(resultater)

counter = 0
for elem in sorted:
    time, layer, count = elem
    path = "c:\\users\\blomst\\best_400x400\\"
    pmang.make_directory(path)

    imagename = str(counter) + "_" + str(time) + "_" + str(layer) + ".png"
    image_path = path + imagename 
    pmang.save_patch_as_image(X[time, layer],image_path)
    counter = counter + 1
exit()

patches = pmang.get_patches(X, rs, window_size)


path_x_train = 'C:\\Users\\Blomst\\dataset\\x_train\\'
path_y_train = 'C:\\Users\\Blomst\\dataset\\y_train\\'



X = []
Y = []

length = len(patches)
count = 1
for patch in patches:
    #removed_row = pmang.set_rows_zero(patch, 7, 32) #Only keep every 8'th row dataset 1

    removed_row = pmang.set_rows_zero(patch, 4, 32) #Only keep every 6'th row dataset 0

    path_x_train_full = path_x_train + str(count) + ".png"
    path_y_train_full = path_y_train + str(count) + ".png"

    patch = pmang.vec_2_patch(patch, 32)
    removed_row = pmang.vec_2_patch(removed_row, 32)


    Y.append(grayscale_2_rgb(patch))
    X.append(grayscale_2_rgb(removed_row))

    count = count + 1

    if(count%1000 == 0): print(count , "/", length )


final = np.asarray([X,Y])

print(np.shape(final))
print(np.shape(final[0]))
print(np.shape(final[1]))

np.save("nn_dataset_348x348x12_random_200k.npy", final)
#data = np.load("nn_dataset.npy")