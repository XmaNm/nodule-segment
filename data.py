from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import cv2

class dataProcess(object):
    def __init__(self,out_rows,out_cols,data_path = "augin/",mask_path = "augout",
                 test_path = "G_input/test/",nodule_path = "D_input/train/",npy_path = "npydata/",
                 res_path = "results/",img_type = "jpg"):

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.mask_path = mask_path
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path
        self.nodule_path = nodule_path
        self.res_path =res_path

    def create_train_data(self):
        i = 0
        j = 0
        print('*'*30)
        print('creating training images...')
        print('*'*30)

        imgs = os.listdir(self.data_path)
        msks = os.listdir(self.mask_path)
        print "numbers of train images", len(imgs)
        print "numbers of train masks", len(msks)
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        imgmasks = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)

        for imgname in imgs:
            img = cv2.imread(os.path.join(self.data_path, imgname), cv2.IMREAD_GRAYSCALE)
            img = img_to_array(img)
            imgdatas[i] = img
            i += 1

        for maskname in msks:
            mask = cv2.imread(os.path.join(self.mask_path, maskname), cv2.IMREAD_GRAYSCALE)
            mask = img_to_array(mask)
            imgmasks[j] = mask
            j += 1

        print('loading done')
        np.save(self.npy_path + 'imgs_train.npy', imgdatas)
        np.save(self.npy_path + 'imgs_mask_train.npy', imgmasks)
        print('Saving to .npy files done.')
        print(' ' * 30)

    def create_test_date(self):
        i = 0
        print('*'*30)
        print('Creating test images...')
        print('*'*30)

        imgs = os.listdir(self.test_path)
        print "numbers of test images:", len(imgs)
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols,1),dtype=np.uint8)
        for imgname in imgs:
            img = cv2.imread(os.path.join(self.test_path, imgname), cv2.IMREAD_GRAYSCALE)
            img = img_to_array(img)
            imgdatas[i] = img
            i += 1
        print('loading done')
        np.save(self.npy_path + 'imgs_test.npy', imgdatas)
        print('Saving to imgs_test.npy files done.')
        print(' '*30)

    def create_nodule_data(self):
        k = 0
        print('*'*30)
        print('Creating nodule images...')
        print('*'*30)
        nods = os.listdir(self.nodule_path)
        print "numbers of nodule images:", len(nods)
        noddatas = np.ndarray((len(nods),40,40,1),dtype=np.uint8)
        for nodname in nods:
            nod = cv2.imread(os.path.join(self.nodule_path,nodname), cv2.IMREAD_GRAYSCALE)
            nod = cv2.resize(nod, (40, 40), interpolation = cv2.INTER_CUBIC)
            nod = img_to_array(nod)
            noddatas[k] = nod
            k += 1
        print('loading done')
        np.save(self.npy_path + 'nodule.npy', noddatas)
        print('Saving to nodule.npy files done.')
        print('*' * 30)

    def load_train_data(self):
        print('*'*30)
        print('load train images...')
        print('*'*30)
        imgs_train = np.load(self.npy_path + "imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path + "imgs_mask_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        imgs_mask_train /= 255
        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0
        return  imgs_train,imgs_mask_train

    def load_test_data(self):
        print('*' * 30)
        print('load test images...')
        print('*' * 30)
        imgs_test = np.load(self.npy_path + "imgs_test.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        return imgs_test

    def load_nodule_data(self):
        print('*'*30)
        print('load nodule images...')
        print('*'*30)
        nodule_imgs = np.load(self.npy_path + "nodule.npy")
        res_imgs = np.load(self.res_path + "nodule.npy")
        nodule_imgs = nodule_imgs.astype('float32')
        nodule_imgs /= 255
        res_imgs = res_imgs.astype('float32')
        res_imgs /= 255
        return nodule_imgs,res_imgs

if __name__== "__main__":

    # aug = myAugmentation()
    # aug.Augmentation()
    mydata = dataProcess(360,360)
    mydata.create_train_data()
    # mydata.create_test_date()
    # mydata.create_nodule_data()






























