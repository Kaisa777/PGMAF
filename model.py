"""
Name: model
Date: 2024/7
"""
import torch.nn as nn
# import torch.nn.modules as nn
import torch.nn.functional as F
import torchvision.models as cv_models
import torch
import os
import math
#import matplotlib.pyplot as plt
import copy
import numpy as np
from transformers import BertModel, BertTokenizer

#自适应多模态融合
class Fusion(nn.Module):
    def __init__(self, fusion_dim):
        super(Fusion, self).__init__()

        self.fc1 = nn.Linear(fusion_dim, 8192)
        self.BN1 = nn.BatchNorm1d(8192)
        self.act1 = nn.ReLU()

        self.fc2 = nn.Linear(8192, 4096)
        self.BN2 = nn.BatchNorm1d(4096)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(4096, fusion_dim)
        self.BN3 = nn.BatchNorm1d(fusion_dim)
        self.act3 = nn.ReLU()

    def forward(self, f1):
        f1 = self.fc1(f1)
        f1 = self.BN1(f1)
        f1 = self.act1(f1)

        f2 = self.fc2(f1)
        f2 = self.BN2(f2)
        f2 = self.act2(f2)

        f2 = self.fc3(f2)
        f2 = self.BN3(f2)
        f2 = self.act3(f2)

        return f1, f2


class ModelParam:
    def __init__(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None, segment_token=None, image_coordinate_position_token=None):
        self.texts = texts
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask
        self.segment_token = segment_token
        self.image_coordinate_position_token = image_coordinate_position_token

    def set_data_param(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None, segment_token=None, image_coordinate_position_token=None):
        self.texts = texts
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask
        self.segment_token = segment_token
        self.image_coordinate_position_token = image_coordinate_position_token

#对比学习
class pllearning(nn.Module):
    def __init__(self, opt):
        super(pllearning, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.temperature = opt.temperature

    def forward(self, feature, prototypes, labels):
        # 归一化 feature 和 prototypes
        feature = F.normalize(feature)
        prototypes = F.normalize(prototypes)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(feature, prototypes.T) / self.temperature

        # 计算 prototype learning 损失
        pl_loss = self.ce(similarity_matrix, labels)
        return pl_loss

#融合
class FuseModel(nn.Module):
    def __init__(self, opt):
        super(FuseModel, self).__init__()

        self.img_hidden_dim = opt.img_hidden_dim
        self.txt_hidden_dim = opt.txt_hidden_dim
        self.common_dim = 256
        # self.common_dim = opt.img_hidden_dim[-1]
        self.classes = opt.classes
        self.batch_size = opt.batch_size
        self.alpha = opt.alpha
        self.beta = opt.beta

        assert self.img_hidden_dim[-1] == self.txt_hidden_dim[-1]
        params = torch.ones(2, requires_grad=True)
        self.params = torch.nn.Parameter(params)

        self.dropout = opt.dropout
        self.fusionnn = Fusion(fusion_dim=self.common_dim)

        # CNN 模型
        self.cnn = cv_models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2]) 
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_fc = nn.Linear(2048, self.common_dim)

        # BERT 模型
        self.bert = BertModel.from_pretrained('../bert-base-uncased')
        self.text_fc = nn.Linear(self.bert.config.hidden_size, self.common_dim)

        self.ifeat_gate = nn.Sequential(
            nn.Linear(self.common_dim, self.common_dim * 2),
            nn.ReLU(),
            nn.Linear(self.common_dim * 2, self.common_dim),
            nn.Sigmoid()
        )
        
        self.tfeat_gate = nn.Sequential(
            nn.Linear(self.common_dim, self.common_dim * 2),
            nn.ReLU(),
            nn.Linear(self.common_dim * 2, self.common_dim),
            nn.Sigmoid()
        )
        self.activation = nn.ReLU()
        self.neck = nn.Sequential(
            nn.Linear(self.common_dim, self.common_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(self.common_dim * 4, self.common_dim)
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(self.common_dim, self.common_dim),
            nn.ReLU()
        )


    def forward(self, text, attention_mask, images):
        self.batch_size = len(images)

        # 提取图像特征
        image_features = self.cnn(images)
        image_features = self.pool(image_features)
        image_features = image_features.view(image_features.size(0), -1)
        imageH = self.image_fc(image_features)

        # 提取文本特征
        outputs = self.bert(text, attention_mask=attention_mask)
        # text_features = outputs.last_hidden_state
        text_features = outputs.pooler_output
        textH = self.text_fc(text_features)

        ifeat_info = self.ifeat_gate(imageH)
        tfeat_info = self.tfeat_gate(textH)
        image_feat = ifeat_info * imageH
        text_feat = tfeat_info * textH


        # fused_features = self.alpha * imageH + self.beta * textH
        fused_features = self.alpha * torch.mul(image_feat, self.params[0].unsqueeze(0)) + \
                         self.beta * torch.mul(text_feat, self.params[1].unsqueeze(0))

        _, fused_fine = self.fusionnn(fused_features)
        cfeat_concat = self.fusion_layer(fused_fine)
        cfeat_concat = self.activation(cfeat_concat)
        nec_vec = self.neck(cfeat_concat)     
        
        # return fused_features, textH, imageH
        return nec_vec, textH, imageH
