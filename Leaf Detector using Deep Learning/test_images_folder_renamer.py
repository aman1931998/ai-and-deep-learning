import os
import shutil

l = os.listdir('test_images')
l.remove('renamer.py')
l.remove('test_images')

count = 0
for i in l:
    shutil.copyfile(os.path.join('test_images', i), os.path.join('test_images', 'test_images', str(count)+'.jpg'))
    count+=1
