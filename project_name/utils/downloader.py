import kagglehub
import os
import shutil

path = kagglehub.dataset_download("orvile/brain-cancer-mri-dataset")
destination_folder = "./data/"

os.mkdir(destination_folder)
shutil.copy(os.path.join(path, 'dataset.csv'), os.path.join(destination_folder, 'dataset.csv'))

for filename in os.listdir(os.path.join(path, 'Brain_Cancer raw MRI data/Brain_Cancer')):
    os.mkdir(os.path.join(destination_folder, filename))
    for image in os.listdir(os.path.join(path, 'Brain_Cancer raw MRI data/Brain_Cancer', filename)):
        shutil.copy(os.path.join(path, 'Brain_Cancer raw MRI data/Brain_Cancer', filename, image), os.path.join(destination_folder, filename, image))
    


print(os.listdir(path))