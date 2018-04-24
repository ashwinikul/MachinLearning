# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 14:10:24 2017

@author: Ashwini
"""

from PIL import Image
import os, os.path
from resizeimage import resizeimage 

imgs = []
path = "/Python/HW12/images/"
output = "/Python/HW12/img_out/"
valid_images = [".jpg",".gif",".png",".tga"]



for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    
    if ext.lower() not in valid_images:
        continue
    image = Image.open(os.path.join(path,f)).convert('L')
    
    if (image.width > image.height):
        image = resizeimage.resize_height(image,52)
        flag = 1
    else:
        flag = 2
        image = resizeimage.resize_width(image,52)
    
    imgs.append(image)
    name= os.path.splitext(f)[0]
    #image.save(os.path.join(output,f))
    
    scale= max(image.size)
    scale = ((scale-52)//5)+1
   

    mean_values=[]

    for y in range(scale):
        
        if (flag == 1):
            cropped_image=image.crop((5*y,0,52+5*y,52))         # wo_ext = name.split(".")
        else:
            cropped_image=image.crop((0,5*y,52,52+5*y)) 
        newFile = name + "_processed_"+ str(y) +".jpg"
        cropped_image.save(os.path.join(output,newFile))
        

        
print("Output images at Path :", output)        
    
    
