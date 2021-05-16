from os import listdir, remove 
from zipfile import ZipFile

files_in_dir = listdir()
np_files = [x for x in files_in_dir if x.endswith('.npy')]

with ZipFile('zipped_npy.zip', 'w') as zipObj2:
    for filename_ in np_files:
        zipObj2.write(filename_)
        remove(filename_)


