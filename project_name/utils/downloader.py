
import kagglehub
import os
import shutil

path = kagglehub.dataset_download("orvile/brain-cancer-mri-dataset")
# Adjust destination_folder to be relative to the script's location, pointing to project_name/../data
destination_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

shutil.copy(os.path.join(path, 'dataset.csv'), os.path.join(destination_folder, 'dataset.csv'))

for filename in os.listdir(os.path.join(path, 'Brain_Cancer raw MRI data/Brain_Cancer')):
    os.mkdir(os.path.join(destination_folder, filename))
    for image in os.listdir(os.path.join(path, 'Brain_Cancer raw MRI data/Brain_Cancer', filename)):
        shutil.copy(os.path.join(path, 'Brain_Cancer raw MRI data/Brain_Cancer', filename, image), os.path.join(destination_folder, filename, image))
    


print(os.listdir(path))