import torch
import torch.nn as nn
from torchvision.ops import nms
import numpy as np

anchor_shape=torch.tensor([[[116,90],[156,198],[373,326]],
                            [[30,61],[62,45],[59,119]],
                            [[10,13],[16,30],[33,23]]], requires_grad=False)
anchor_shape = anchor_shape.cuda()

class Anchor_Box():
    def __init__(self, anchor_shape=anchor_shape):
        self.anchor_shape = anchor_shape
    def Box_Decode(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):
            #-----------------------------------------------#
            #   输入的inputs一共有三个，他们的shape分别是
            #   batch_size, 255, 13, 13
            #   batch_size, 255, 26, 26
            #   batch_size, 255, 52, 52
            #-----------------------------------------------#
            batch_size = input.size(0)
            input_height = input.size(2)
            input_width = input.size(3)
            prediction = input.reshape((batch_size, 3 , 85, input_height, input_width))
            
            x = torch.sigmoid(prediction[:,:,0,:,:]) # x.shape = (batch_size,3,input_height,input_width)
            y = torch.sigmoid(prediction[:,:,1,:,:])
            w = torch.sigmoid(prediction[:,:,2,:,:])
            h = torch.sigmoid(prediction[:,:,3,:,:])
            x = x.cuda()
            y = y.cuda()
            w = w.cuda()
            h = h.cuda()

            objectnesss = torch.sigmoid(prediction[:,:,4,:,:])
            pred_cls = torch.sigmoid(prediction[:,:,5:,:,:]) # x.shape = (batch_size,3,80,input_height,input_width)
            objectnesss = objectnesss.cuda()
            pred_cls = pred_cls.cuda()
            #----------------------------------------------------------#
            #   生成网格，先验框中心，网格左上角 
            #   batch_size,3,input_height,input_width
            #----------------------------------------------------------#
            grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_width, 1).repeat(3,1,1).repeat(batch_size,1,1,1).type(torch.cuda.FloatTensor)
            grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_height, 1).t().repeat(3,1,1).repeat(batch_size,1,1,1).type(torch.cuda.FloatTensor)
            #-----------------------------------------------------------#
            #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
            #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
            #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
            #-----------------------------------------------------------#
            stride_w = 416/input_width
            stride_h = 416/input_height
            
            anchor_w = self.anchor_shape[i,:,0]
            anchor_w = anchor_w.reshape(3,1,1).repeat(1,input_width,input_height).repeat(batch_size,1,1,1)/stride_w
            
            anchor_h = self.anchor_shape[i,:,1]
            anchor_h = anchor_h.reshape(3,1,1).repeat(1,input_width,input_height).repeat(batch_size,1,1,1)/stride_h

            pred_box = torch.cuda.FloatTensor(prediction[:,:,0:4,:,:].shape)
            pred_box[:,:,0,:,:] = x.detach() + grid_x
            pred_box[:,:,1,:,:] = y.detach() + grid_y
            pred_box[:,:,2,:,:] = torch.exp(w.detach()) * anchor_w
            pred_box[:,:,3,:,:] = torch.exp(h.detach()) * anchor_h
            
            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(torch.cuda.FloatTensor)
            output = torch.cat((pred_box.view(batch_size, -1, 4) / _scale,
                                objectnesss.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, 80)), -1)
            outputs.append(output.detach())
        return outputs
