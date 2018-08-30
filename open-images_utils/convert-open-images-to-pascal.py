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
print ('complete read csv: '+str(your_list.shape[0])+' objects')

    
in_person={'/m/04yx4': 'male','/m/03bt1vf':'female','/m/01bl7v':'boy','/m/05r655':'girl'}





for i in range(1,your_list.shape[0]):   
    

    imgname=your_list[i,0]
    






    #print('{}/{}       {}'.format(i,your_list.shape[0],imgname))
    
    
    classname=your_list[i,2]
    if (classname == '/m/04yx4') or (classname == '/m/03bt1vf') or (classname == '/m/01bl7v') or (classname == '/m/05r655'):
        good_img=1
        
        im = Image.open('../dataset/train_images/'+your_list[i,0]+'.jpg')
        img_x, img_y= im.size

        xmin=round(float(your_list[i,4])*img_x)
        xmax=round(float(your_list[i,5])*img_x)
        ymin=round(float(your_list[i,6])*img_y)
        ymax=round(float(your_list[i,7])*img_y)


        #
        if xmax-xmin > img_x/2:
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$${}'.format(imgname))
        if ymax-ymin > img_y/2:
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$${}'.format(imgname))

  
        writer=pasc.Writer(your_list[i,0]+'.jpg',img_x,img_y)
        writer.addObject(in_person[classname], xmin, ymin, xmax, ymax)   
        
    

    print(good_img)
    if good_img==1:
        
        if your_list[i+1,0] != imgname:
            writer.save('xml_ann/'+imgname+'.xml')
           # print('                               saved : '+imgname+'.xml')

            good_img=0
            
            




if good_img==1:
    writer.save('xml_ann/'+imgname+'.xml')
    #print('                               saved last: '+imgname+'.xml')









