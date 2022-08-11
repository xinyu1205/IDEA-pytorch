
# --------------------------------------------------------
# CLIP & IDEA
# Written by Xinyu Huang
# --------------------------------------------------------


import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from transformers import BertModel, BertConfig
from dataset.class_config import class_array, class_weight
from models.mldecoder import MLDecoder


class CLIP(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,    
                 temp = 0.07,
                 init_deit = True
                 ):
        super().__init__()
        
        # Image Encoder config
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)          
        vision_width = config['vision_width'] 

        # Text Encoder config
        self.tokenizer = tokenizer 
        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertModel.from_pretrained(text_encoder,config=bert_config)     
        text_width = self.text_encoder.config.hidden_size

        # ITC config
        embed_dim = config['embed_dim']
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         
        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   


    def forward(self, image, text, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        
        image_embeds = self.visual_encoder(image) 
        
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  

        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask, return_dict = True)            
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                 
             
        sim_i2t = image_feat @ text_feat.t() / self.temp 
        sim_t2i = text_feat @ image_feat.t() / self.temp 

        with torch.no_grad():
            sim_targets = torch.zeros(sim_i2t.size()).to(image.device)
            sim_targets.fill_diagonal_(1)
                             
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean() 

        loss_itc = (loss_i2t+loss_t2i)/2
        
        return loss_itc 


class IDEA(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,    
                 temp = 0.07,
                 init_deit = True
                 ):
        super().__init__()

        # Image Encoder config
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)          
        vision_width = config['vision_width'] 

        # Text Encoder config
        self.tokenizer = tokenizer 
        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertModel.from_pretrained(text_encoder,config=bert_config)   
        text_width = self.text_encoder.config.hidden_size

        # MLR config
        num_class = config['class_num']
        self.ml_decoder = MLDecoder(num_classes=num_class, initial_num_features=vision_width)
        self.class_weight = class_weight
        self.tau = config['tau']
        self.change_epoch = config['change_epoch']

        # Tag2Text config
        self.class_array = class_array

        # ITC config
        embed_dim = config['embed_dim']
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         
        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   


    def forward(self, image, text, tag, epoch=0, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        
        # get image spatial feature and global feature
        image_embeds = self.visual_encoder(image) 
        image_spatial_feature = image_embeds[:,1:,:]
        image_global_feature = image_embeds[:,0,:]

        # Multi-Label Recognition
        l_mlr, tag, pseudo = self.mlr(image_spatial_feature, tag, epoch)

        # conver tag to text
        tag2text = self.Merge(tag)
        text.extend(tag2text)

        # get text global feature
        text_input = self.tokenizer(text, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(torch.device('cuda'))
        text_output = self.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, return_dict = True)       
        text_global_feature = text_output.last_hidden_state[:,0,:]
        
        # Image Text Contrastive Learning
        l_itc = self.itc(image_global_feature, text_global_feature, pseudo)

        return l_itc/l_itc.detach(), l_mlr/l_mlr.detach()
    
    def mlr(self, xs, tag, epoch):
        logits = self.ml_decoder(xs)

        with torch.no_grad():
            targets = tag 
            pseudo = torch.zeros(xs.shape[0]).cuda()

        if epoch >= self.change_epoch:
            targets = torch.where(torch.sigmoid(logits) > self.tau, torch.tensor(1).cuda(), targets)
        
        l_mlr = F.multilabel_soft_margin_loss(logits, targets, weight=self.class_weight.cuda(),reduction='mean')

        pseudo = torch.where((targets - tag).sum(dim = 1) > 0, torch.tensor(1.0).cuda(), pseudo).unsqueeze(-1)

        return l_mlr, targets * pseudo, pseudo

    def itc(self, image_global_feature, text_global_feature, pseudo ):
        zi = F.normalize(self.vision_proj(image_global_feature), dim = -1)
        zt = F.normalize(self.text_proj(text_global_feature), dim = -1)
        btc = image_global_feature.shape[0]

        similarity_i2t = zi @ zt.t() / self.temp 
        similarity_t2i = zt @ zi.t() / self.temp
        
        with torch.no_grad():
            sim_targets = torch.zeros(btc,btc).to(torch.device('cuda')).fill_diagonal_(1)
            sim_i2t_targets = torch.cat((sim_targets, sim_targets * pseudo),1)
            sim_t2i_targets = torch.cat((sim_targets, sim_targets * pseudo),0)

        loss_i2t = -torch.sum(F.log_softmax(similarity_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(similarity_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        l_itc = (loss_i2t+loss_t2i)/2

        return l_itc

    def Merge(self, tag):
        tag = tag.cpu().numpy()

        tag2text = []
        for i in range(tag.shape[0]):
            index = np.argwhere(tag[i]==1)
            token = self.class_array[index].squeeze(axis = 1)
            tag2text.append(' '.join(token))

        return tag2text

