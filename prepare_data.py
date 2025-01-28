import os
import shutil

root_dir = 'animals'
selected_data_dir = 'selected_data_animals'

if not os.path.exists(selected_data_dir):
    os.makedirs(selected_data_dir)

for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if os.path.isdir(folder_path):
        images = os.listdir(folder_path)
        for i, image in enumerate(images[:40]):
            image_path = os.path.join(folder_path, image)
            shutil.copy(image_path, os.path.join(selected_data_dir, f'{folder}_{i + 1}.jpg'))
