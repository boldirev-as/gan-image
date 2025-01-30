import os
import shutil

for root, _, files in os.walk('food_data/test/'):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            shutil.copy(os.path.join(root, file), os.path.join('food_data_test/', file))
