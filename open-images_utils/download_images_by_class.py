# USAGE python download_images_by_class.py | gsutil -m cp -I <destination>

import pascal_voc_writer as pasc
import csv
import numpy as np
from PIL import Image

good_img=0

# read file
with open('annotations-human-bbox.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)

your_list=np.array(your_list)
#print ('complete read csv: '+str(your_list.shape[0])+' objects')

    
in_person={'/m/04yx4': 'male','/m/03bt1vf':'female','/m/01bl7v':'male','/m/05r655':'female'}



#loop

#your_list.shape[0]
mylist = []

for i in range(1,your_list.shape[0]):   
    

    imgname=your_list[i,0]
    





    classname=your_list[i,2]
    if (classname == '/m/04yx4') or (classname == '/m/03bt1vf') or (classname == '/m/01bl7v') or (classname == '/m/05r655'):
      
        good_img=1





    if good_img==1:
        
        if your_list[i+1,0] != imgname:

            print('gs://open-images-dataset/test/'+your_list[i,0]+'.jpg')

            good_img=0
            
            


