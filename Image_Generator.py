import cv2
import os, random
import numpy as np
from parameter import letters

# # Input data generator
def labels_to_text(labels):     # letters index -> text (string)
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))


class TextImageGenerator:
    def __init__(self, img_dirpath,df, img_w, img_h,
                 batch_size, downsample_factor, max_text_len=7):
        self.img_h = img_h
        self.img_w = img_w
        self.df = df
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        self.img_dirpath = img_dirpath                  # image dir path
        self.img_dir = os.listdir(self.img_dirpath)     # images list
        self.n = self.df.shape[0]                      # number of images
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []


    def build_data(self, df):
        img_names = self.df['name'].values
        labels = self.df['label'].values
        print(self.n, " Image Loading start...")
        for i in range(len(img_names)):
            img = cv2.imread(self.img_dirpath + img_names[i], cv2.IMREAD_GRAYSCALE)
    
            try:
                img = cv2.resize(img, (self.img_w, self.img_h))
                img = img.astype(np.float32)
            except:
                print(self.img_dirpath + img_names[i])
                exit()

            img = (img/255.0)
            
            self.imgs[i, :, :] = img
            self.texts.append(labels[i])
        print(len(self.texts) == self.n)
        print(self.n, " Image Loading finish...")
        print(self.texts)

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def next_batch(self):       ## batch size
        while True:
            X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])     # (bs, 128, 64, 1)
            Y_data = np.ones([self.batch_size, self.max_text_len])             # (bs, 9)
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)  # (bs, 1)
            label_length = np.zeros((self.batch_size, 1))           # (bs, 1)

            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                img = np.expand_dims(img, -1)
                X_data[i] = img
                Y_data[i] = text_to_labels(text)
                label_length[i] = len(text)

            inputs = {
                'the_input': X_data,  # (bs, 128, 64, 1)
                'the_labels': Y_data,  # (bs, 8)
                'input_length': input_length,
                'label_length': label_length
            }
            outputs = {'ctc': np.zeros([self.batch_size])} 
            yield (inputs, outputs)
