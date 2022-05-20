import numpy as np
from sklearn.metrics import confusion_matrix

gt=np.load('/Users/kabbas570gmail.com/Documents/Challenge/valid/gt/train_51_39_gt.npy')
p=np.load('/Users/kabbas570gmail.com/Documents/Challenge/valid/gt/train_51_39_gt.npy')

a1=p[0,0,:,:]
b1=gt[0,0,:,:]


def Evaluation_Metrics(pre,gt):
    pre=pre.flatten() 
    gt=gt.flatten()  
    tn, fp, fn, tp=confusion_matrix(gt,pre,labels=[0,1]).ravel()
    
    iou=tp/(tp+fn+fp) 
    dice=2*tp / (2*tp + fp + fn)
    
    print('TPs :',tp)
    print('TNs :',tn)
    print('FNs :',fn)
    print('FPs :',fp)

    return iou,dice



import torch
gt=torch.tensor([0,1,1,0,1,0])

iou_,dice_=Evaluation_Metrics(gt,gt)



a=np.array([[1,1,0],[1,1,1]])
b=np.array([[1,1,0],[1,1,1]])

a = np.expand_dims(a, 0)
pre=a.flatten() 

iou_,dice_=Evaluation_Metrics(a,b)

a=a.view(-1)
def iou_f(inputs, targets, smooth=1):
    
    #comment out if your model contains a sigmoid or equivalent activation layer
    #inputs = F.sigmoid(inputs)       
    
    #flatten label and prediction tensors
    inputs = inputs.flatten()
    targets = targets.flatten()
    
    #intersection is equivalent to True Positive count
    #union is the mutually inclusive area of all labels & predictions 
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection 
    
    IoU = (intersection + smooth)/(union + smooth)
            
    return IoU
iou_1=iou_f(a,b)


from torchmetrics import JaccardIndex
jaccard = JaccardIndex(num_classes=2)
import torch
a_ = torch.tensor(a)
b_= torch.tensor(b)


def iou_f(inputs, targets, smooth=1):
    
    inputs=inputs
    targets=targets
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection 
    IoU = (intersection + smooth)/(union + smooth)      
    return IoU

iou_2=iou_f(a_,b_)

print(iou_2)
J=jaccard(a_, b_)
print(J)
f1=a.flatten()
f2=b.flatten()



intersection = np.logical_and(a, b)
union = np.logical_or(a, b)
iou_score = np.sum(intersection) / np.sum(union)





