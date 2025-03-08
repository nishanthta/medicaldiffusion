import shutil, os

for file in os.listdir('../MAYO_3D/all'):
    if 'liver' in file:
        os.remove(os.path.join('../MAYO_3D/all', file))