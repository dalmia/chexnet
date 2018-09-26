import os
import numpy as np
import time
import sys
from PIL import Image

import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from DensenetModels import DenseNet121
from DensenetModels import DenseNet169
from DensenetModels import DenseNet201

#-------------------------------------------------------------------------------- 
#---- Class to generate heatmaps (CAM)

class HeatmapGenerator ():
    
    #---- Initialize heatmap generator
    #---- pathModel - path to the trained densenet model
    #---- nnArchitecture - architecture name DENSE-NET121, DENSE-NET169, DENSE-NET201
    #---- nnClassCount - class count, 14 for chxray-14

 
    def __init__ (self, pathModel, nnArchitecture, nnClassCount, transCrop):
       
        #---- Initialize the network
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, True).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, True).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, True).cuda()
          
        model = torch.nn.DataParallel(model).cuda()

        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'])
        
        self.feature_extractor = model.module.densenet121.features
        self.feature_extractor.eval()
        self.classifier = model.module.densenet121.classifier
        
        #---- Initialize the weights
        self.weights = list(self.classifier.parameters())[-2].cpu().data.numpy()
        self.bias = list(self.classifier.parameters())[-1].cpu().data.numpy()

        #---- Initialize the image transform - resize + normalize
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize(transCrop))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        
        self.transformSequence = transforms.Compose(transformList)
    
    #--------------------------------------------------------------------------------
     
    def generate (self, pathImageFile, pathOutputFile, classIndex, classLabel, transCrop):
        
        #---- Load image, transform, convert 
        imageData = Image.open(pathImageFile).convert('RGB')
        imageData = self.transformSequence(imageData)
        imageData = imageData.unsqueeze_(0)
        
        input = torch.autograd.Variable(imageData)
        
        self.feature_extractor.cuda()
        output = self.feature_extractor(input.cuda())
        output = output.squeeze(0)
        output = output.cpu().data.numpy()
        
        #---- Generate heatmap
        heatmap = np.zeros((output.shape[-2], output.shape[-1]))
        for j in range(self.weights.shape[1]):
            heatmap += self.weights[classIndex][j] * output[j]

        heatmap += self.bias[classIndex]
        
        # code below is taken from https://github.com/jrzech/reproduce-chexnet  
        heatmap=1 / (1 + np.exp(-heatmap))
    
        # calculated using frequentist approach
        # based on prevalance of label throughout the dataset
        label_baseline_probs={
            'Atelectasis':0.103,
            'Cardiomegaly':0.025,
            'Effusion':0.119,
            'Infiltration':0.177,
            'Mass':0.051,
            'Nodule':0.056,
            'Pneumonia':0.012,
            'Pneumothorax':0.047,
            'Consolidation':0.042,
            'Edema':0.021,
            'Emphysema':0.022,
            'Fibrosis':0.015,
            'Pleural_Thickening':0.03,
            'Hernia':0.002
        }
        
        #normalize by baseline probabilities
        heatmap = heatmap / label_baseline_probs[classLabel]
        
        #---- Blend original and heatmap 

        imgOriginal = cv2.imread(pathImageFile, 1)
        imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))
        
        cam = (heatmap - np.min(heatmap)) / np.max(heatmap)
        cam = cv2.resize(cam, (transCrop, transCrop))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
              
        img = heatmap * 0.5 + imgOriginal
            
        cv2.imwrite(pathOutputFile, img)
        
#-------------------------------------------------------------------------------- 

pathInputImage = 'test/00011997_003.png'
classIndex = 2
classLabel = 'Effusion'
pathOutputImage = 'test/heatmap.png'
pathModel = 'models/m-25012018-123527.pth.tar'

nnArchitecture = 'DENSE-NET-121'
nnClassCount = 14

transCrop = 224

h = HeatmapGenerator(pathModel, nnArchitecture, nnClassCount, transCrop)
h.generate(pathInputImage, pathOutputImage, classIndex, classLabel, transCrop)
