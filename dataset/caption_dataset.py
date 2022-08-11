import json
import os
import random

from torch.utils.data import Dataset
import numpy as np
import torch

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption
      

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
      
        image = Image.open(ann['image']).convert('RGB')   
        image = self.transform(image)
                
        return image, caption
            

class pretrain_dataset_tag(Dataset):
    def __init__(self, ann_file, transform, max_words=30,class_num = 473):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words

        self.class_num = class_num
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)

        num = ann['tag']
        # num = np.array([int(i) for i in num])
        label = np.zeros([self.class_num])
        label[num] = 1
        label = torch.tensor(label, dtype = torch.long)

        if os.path.exists(ann['image'])==False:
            caption = pre_caption('',self.max_words)
            label = np.zeros([self.class_num])
            label = torch.tensor(label, dtype = torch.long)
            image = np.zeros((256,256,3))
            image = Image.fromarray(np.uint8(image))
            image = self.transform(image)
        else:
            image = Image.open(ann['image']).convert('RGB')
            image = self.transform(image)

        return image, caption, label
            

