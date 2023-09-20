from PIL import Image
import os
import numpy as np
import json
from dataset import DatasetSplit, DatasetRegistry

__all__ = ['register_leaves']

image_id_dict = {}

def create_sub_masks(mask_image):
    width, height = mask_image.size

    sub_masks = {}
    for x in range(width):
        for y in range(height):
            pixel = mask_image.getpixel((x,y))

            if pixel != (0):
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks

class LeafDemo(DatasetSplit):
    def __init__(self):
        pass
    def training_roidbs(self):
        roidbs = []
        path = r'C:\Users\aman1\Downloads\Compressed\CVPPP2017_LSC_training\CVPPP2017_LSC_training\training'
        for sub_path, _, files in list(os.walk(path))[1:]: #sub_path, _, files = list(os.walk(path))[1:][0]
            rgbfiles = [i for i in files if i.endswith('rgb.png')]
            labelfiles = [i for i in files if i.endswith('label.png')]
            
            for i in range(len(rgbfiles)): #i =4
                data = {'file_name': os.path.join(sub_path, rgbfiles[i]), 
                        }
                img_mask = Image.open(os.path.join(sub_path, labelfiles[i]))
                img_mask = create_sub_masks(img_mask)
                
                elements = len(img_mask)
                data['class'] = [1]*elements
                data['is_crowd'] = [False]*elements
                segmentation, boxes = [], []
                
                for j in img_mask.values(): #j = list(img_mask.values())[0]
                    mask = np.asarray(j)
                    width, height = np.max(mask, axis = 0), np.max(mask, axis = 1)
                    for find in range(len(width)):
                        if width[find]: y2 = find + 1
                    for find in range(len(width)):
                        if width[::-1][find]: y1 = len(width) - find - 1
                    for find in range(len(height)):
                        if height[find]: x2 = find + 1
                    for find in range(len(height)):
                        if height[::-1][find]: x1 = len(height) - find - 1
                    boxes.append([x1, x2, y1, y2])
                    segmentation.append(np.array(mask, 'uint8')*255)
                data['boxes'] = boxes
                data['segmentation'] = segmentation
                
                roidbs.append(data)
        return roidbs
    
    def inference_roidbs(self):
        global image_id_dict
        roidbs = []
        path = r'C:\Users\aman1\Downloads\Compressed\CVPPP2017_LSC_training\CVPPP2017_LSC_training\A3'
        
        rgbfiles = [i for i in os.listdir(path) if i.endswith('rgb.png')]
        
        count = 0
        for i in rgbfiles: #
            roidbs.append({'file_name':os.path.join(path, i), 
                           'image_id':count})
            count+=1
        
        return roidbs
    
    def eval_inference_results()










