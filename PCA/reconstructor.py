import numpy as np
from sklearn.decomposition import PCA
import math
import patch_manager as pmang




#Read tiff


# https://stackoverflow.com/questions/28274091/removing-completely-isolated-cells-from-python-array

class Reconstructor:
    
    Vecs = []
    Eig = []
    mu = []
    got_pca = False
    n_var = 0
    
    
    def get_n_variance(self, pca, variance):
        '''returns the number of components need to have (variance)%'''
        sum = 0.0
        n = 0
        while sum < variance:
            sum = sum + pca.explained_variance_ratio_[n]
            n = n + 1
        return n

    def get_pca(self,patches :np.ndarray, variance=1.0, n_componets = 0):
        '''
            Makes PCA and saves the vectors, eigen values and mean in Reconstructor object
        '''
        pca = PCA(svd_solver="full")
        pca.fit(patches)
        self.Eig = pca.singular_values_
        self.Vecs = pca.components_
        self.mu = pca.mean_
        self.got_pca = True
        self.n_var = self.get_n_variance(pca, 0.95)
        return (self.Vecs , self.Eig, self.mu)

    def learn_const(self,patch):
        '''Computes the constants K'''
        patch_vec = patch.reshape(-1)
        learned_const = []
        center = patch_vec - self.mu
        for i in range(len(self.Eig)):
            v = self.Vecs[i]
            const = np.dot((center), v)
            learned_const.append(const)
        return learned_const

    def create_new_patch(self,constants, n_components):
        '''Creates new patch based on constants'''
        result = self.mu
        for i in range(n_components):
            k = constants[i]
            v = self.Vecs[i]
            result = result + k*v
        return result
    
    def create_new_patch_n1(self,constants, n_component):
        '''Creates new patch based on constants'''
        k = constants[n_component]
        v = self.Vecs[n_component]
        return k*v
    
    def create_new_patch_n2(self, n_component):
        '''Creates new patch based on constants'''
        v = self.Vecs[n_component]
        return v
    
    def reconstruct_patch_n1(self,patch, n_component):
        '''Returns a new array reconstructed from PCA model and constants'''
        if self.got_pca == False:
            raise Exception("PCA not created. Call get_pca before reconstructing")
        else:
            consts = self.learn_const(patch)

            npatch = self.create_new_patch_n1(consts, n_component)
            return npatch

    def reconstruct_patch(self,patch, n_components):
        '''Returns a new array reconstructed from PCA model and constants'''
        if self.got_pca == False:
            raise Exception("PCA not created. Call get_pca before reconstructing")
        else:
            consts = self.learn_const(patch)

            npatch = self.create_new_patch(consts, n_components)
            return npatch
    
    def reconstruct_image(self, img, n_components, window_size, nth):
        win_minus_1 = window_size-1
        X,Y = img.shape
        recon = np.zeros((X,Y))
        for i in range(X-window_size):
            for j in range(Y-window_size):
                patch_row = pmang.get_patch(img, i, j, window_size)
                patch = pmang.vec_2_patch(patch_row, window_size)

                removed  = np.copy(patch)
                for k in range(len(patch)):
                    #((i+k)%32) % 7 
                    if((i+k) % nth != 0):
                        removed = pmang.remove_row_from_patch(removed, k, window_size, self.mu)

                removed_row = pmang.patch_2_vec(removed)
                recon_patch = self.reconstruct_patch(removed_row, n_components)
                x = i + int(win_minus_1/2)
                y = j + int(win_minus_1/2)

                row_index = int(win_minus_1/2)*win_minus_1+int(win_minus_1/2)
                recon[x][y] = recon_patch[row_index]
                #print(j,"/",Y-window_size)
            print(i,"/",X-window_size)
        return recon

    def save_principal_components_as_image (self, k, path, window_size):
        pmang.make_directory(path)
        for i in range(len(self.Eig)):
            patch = pmang.vec_2_patch((self.mu + (k*np.sqrt(self.Eig[i])*self.Vecs[i])), window_size)
            path_i = path + str(i) + ".png"
            pmang.save_patch_as_image(patch, path_i)
        return 0

