import os
import numpy as np
import h5py
from scipy import misc
from PIL import Image
import glob
from matplotlib import pyplot as plt
from skimage.io import imsave, imread
import scipy.io as sio

###################### Paths ######################

############################################



#Path to the images, annotation maps

path_data = 'image/'

raw_img_rows, raw_img_cols = 572, 572
Num_imgs = 1
channels = 3


def train_data():
    # start of by training only on 1 image
    initial_training_path = os.path.join(path_data, 'train_1_image')
    #initial_annot_path = os.path.join(path_data, 'annot_1_image')
    # TODO for the major training we will initialize a different path that contains all of the training set
    #train_path = os.path.join(path_data, 'train_set')

    first_img = os.listdir(initial_training_path)
    #first_annot = os.listdir(initial_annot_path)
    #total = len(first_img)/2

    # creating a 3D matrix with 1 image, with given pixels, first element number of columns, 2nd number of rows, 3rd RGB
    imgs = np.empty((Num_imgs,raw_img_rows,raw_img_cols,channels), dtype=np.uint8)
    imgs_annot = np.empty((Num_imgs, raw_img_rows, raw_img_cols, channels), dtype=np.uint8)


    print('-' * 32)
    print('Creating the training image')
    print('-' * 32)

    i = 0
    for image in first_img:
        # read the image
        img = imread(os.path.join(initial_training_path, image), mode='RGB')
        print(img.shape, img.dtype)
        img = np.array([img])
        # save it in images
        imgs[i] = img



   # for annot_image in first_annot:

        # read the annotated image
    #    annot_img = imread(os.path.join(initial_annot_path, annot_image), mode='RGB')
    #    print(annot_img.shape, annot_img.dtype)
    #    annot_img = np.array([annot_img])
        # save it in annotated images
    #    imgs_annot[i] = annot_img

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, 1))
        i += 1

    print('loading images complete')
    np.save('imgs_train.npy', imgs)
   # np.save('imgs_annot.npy', imgs_annot)
    print('Saving to .npy files done.')

def load_training_set():
    train_set = np.load('imgs_train.npy')

    print(train_set)

    return train_set

train_data()


print(np.load(('imgs_train.npy'))[0].shape)
#print(sio.loadmat('img/annot_1_image/pat19_im1_NDBT.mat'))
#plt.imshow(np.load('imgs_train.npy')[0])
#plt.imshow(np.load('imgs__annot.npy')[0])
#plt.show()
