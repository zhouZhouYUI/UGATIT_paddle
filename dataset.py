from paddle import reader
from PIL import Image

import os
import os.path

import paddle
import paddle.fluid as fluid
import cv2
from utils import *
from PIL import Image, ImageOps
import matplotlib.pylab as plt

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, extensions):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                item = (path, 0)
                images.append(item)

    return images

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)



class Selfie2AnimeDataReader(object):
    def __init__(self, image_path, batch_size=1, img_size=256, img_mean=0.5, img_std=0.5, data_improve=True) :
        self.image_path = image_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_mean = img_mean
        self.img_std = img_std
        self.data_improve = data_improve

    # ----- paddle 数据读取
    def create_data_reader(self, file_dir):

        def reader():
            filename_list = os.listdir(file_dir)
            for img_path in filename_list:
                
                if img_path[img_path.rindex('.'):] in IMG_EXTENSIONS:
                    # print(os.path.join(file_dir, img_path))
                    img = cv2.imread(os.path.join(file_dir, img_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    if self.data_improve:
                        # 随机改变亮暗、对比度和颜色等
                        img = random_distort_cv(img)

                        # 随机水平反转
                        img = random_flip_cv(img)

                        # 重新缩放 来重新裁剪
                        img = cv2.resize(img, ( self.img_size+30,  self.img_size+30))
                        # 在随机位置裁剪给定图像
                        img = random_crop_cv(img, self.img_size)

                    # 归一化
                    out_img = (img / 255.0 - self.img_mean) / self.img_std
                    out_img = out_img.astype('float32').transpose((2, 0, 1))
                    out_img = fluid.dygraph.to_variable(out_img)
                    yield fluid.layers.reshape(out_img, shape=(1, 3, self.img_size, self.img_size)) 
        
        return reader 

    def create_reader(self, is_shuffle=True):

        if is_shuffle:
            train_a_img_generator = paddle.batch( paddle.reader.shuffle(self.create_data_reader(self.image_path), 28), self.batch_size)
        else:
            train_a_img_generator = paddle.batch(self.create_data_reader(self.image_path), self.batch_size)
        

        return train_a_img_generator()

def show_img(img):
    plt.imshow(img)
    plt.show()

def show_img_cv(img):
    cv2.imshow("Display window", img)
    k = cv2.waitKey(0)
    print(k)
