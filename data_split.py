import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
TRAIN_CAT_DIR = '../input/train/cat/'
TRAIN_DOG_DIR = '../input/train/dog/'
VAL_DIR = '../input/validation/'
VAL_SIZE = 0.2

dog_validation_files = [VAL_DIR+name for \
                        name in os.listdir(VAL_DIR) \
                        if 'dog' in name]

cat_validation_files = [VAL_DIR+name for \
                        name in os.listdir(VAL_DIR) \
                        if 'cat' in name]

dog_train_files = [TRAIN_DOG_DIR+name for \
                        name in os.listdir(TRAIN_DOG_DIR)]

cat_train_files = [TRAIN_CAT_DIR+name for \
                        name in os.listdir(TRAIN_CAT_DIR)]

all_dog_files = dog_validation_files + dog_train_files
all_cat_files = cat_validation_files + cat_train_files

print("Total Images:\t{}".format(len(all_dog_files)+len(all_cat_files)))

new_dog_train, new_dog_val, _, _ = train_test_split( \
       all_dog_files, all_dog_files, test_size=VAL_SIZE, random_state=2)

new_cat_train, new_cat_val, _, _ = train_test_split( \
       all_cat_files, all_cat_files, test_size=VAL_SIZE, random_state=2)

new_train_set = new_dog_train + new_cat_train
new_val_set = new_dog_val + new_cat_val

print("Total Images:\t{}".format(len(new_train_set)+len(new_val_set)))

print("\nMoving Images to Train directory..")
for file_path in tqdm(new_train_set):
    filename = os.path.basename(file_path)
    if 'dog' in filename:
        shutil.move(file_path, TRAIN_DOG_DIR+filename)
    else:
        shutil.move(file_path, TRAIN_CAT_DIR+filename)

print("\nMoving Images to Validation directory..")
for file_path in tqdm(new_val_set):
    filename =  os.path.basename(file_path)
    shutil.move(file_path, VAL_DIR+filename)
