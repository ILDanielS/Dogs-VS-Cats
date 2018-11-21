import numpy as np
import matplotlib.pyplot as plt
import os, cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import regularizers
from tqdm import tqdm      # a nice pretty percentage bar for tasks.
from random import shuffle

TEST_DIR = '../input/test/'
IMG_SIZE = 12500
#ROWS = 256
#COLS = 256
ROWS = 64
COLS = 64
CHANNELS = 3



def load_image(file_path, size):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    b,g,r = cv2.split(img)
    rgb_img = cv2.merge([r,g,b])
    return cv2.resize(rgb_img, size)

# All test files
test_image_list = [TEST_DIR+i for i in os.listdir(TEST_DIR)]

print(len(test_image_list))


filepath = 'cat_dog_v4a1'

if filepath in os.listdir():
    print('Loading model from disk..')
    model = load_model(filepath)

print(model.summary())


with open('submission_file2.csv','w') as f:
     f.write('id,label\n')
with open('submission_file2.csv','a') as f:
    for i, image_file in tqdm(enumerate(test_image_list)):
        image = np.expand_dims(load_image(image_file,(ROWS,COLS)), axis=0) / 255.0
        prediction = model.predict(image)
        f.write('{},{}\n'.format(i+1,prediction[0][0]))
