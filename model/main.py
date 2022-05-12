from torch import nn
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import torch.optim as optim
from tqdm import tqdm

import numpy as np

from data_read import load_data

trin_path='/Users/kabbas570gmail.com/Documents/ResNetClassifier/Tumor_Data/Train'
train_loader=load_data(trin_path)


val_path='/Users/kabbas570gmail.com/Documents/ResNetClassifier/Tumor_Data/Train'
val_loader=load_data(val_path)

test_path='/Users/kabbas570gmail.com/Documents/ResNetClassifier/Tumor_Data/Train'
test_loader=load_data(test_path)


print("The length of the Training DataLoader Dataset is: ",len(train_loader))
print("\nThe length of the Testing  DataLoader Dataset is: ",len(val_loader))           
print("\nThe length of the Valid  DataLoader Dataset is: ",len(test_loader)) 


### Loading Model  ###

import timm
model = timm.create_model('resnet50', pretrained=True)
model.fc=nn.Linear(in_features=2048, out_features=3, bias=True)

from torchsummary import summary
model.to(device=DEVICE,dtype=torch.float)
summary(model, (3, 224, 224))


avg_train_losses = []   # losses of all training epochs
avg_valid_losses = []  #losses of all training epochs
avg_valid_DS = []  # all training epochs


NUM_EPOCHS=200
LEARNING_RATE=0.0001

def train_fn(loader_train,loader_valid, model, optimizer, loss_fn1, scaler):
    
     
    train_losses = [] # loss of each batch
    valid_losses = []  # loss of each batch

    
    loop = tqdm(loader_train)
    for batch_idx, (data, label) in enumerate(loop):
        data = data.to(device=DEVICE,dtype=torch.float)
        label = label.to(device=DEVICE,dtype=torch.float)
        # forward
        with torch.cuda.amp.autocast():
            out1 = model(data)    
            loss1 = loss_fn1(out1, label)
        # backward
        loss=loss1
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        train_losses.append(loss.item())
    
    loop_v = tqdm(loader_valid)
    model.eval()
    for batch_idx, (data, label) in enumerate(loop_v):
        data = data.to(device=DEVICE,dtype=torch.float)
        label = label.to(device=DEVICE,dtype=torch.float)
        # forward
        with torch.cuda.amp.autocast():
            out1 = model(data)    
            loss1 = loss_fn1(out1, label)
        loop_v.set_postfix(loss=loss.item())
        valid_losses.append(loss.item())
    model.train()
    
    train_loss_per_epoch = np.average(train_losses)
    valid_loss_per_epoch = np.average(valid_losses)
    ## all epochs
    avg_train_losses.append(train_loss_per_epoch)
    avg_valid_losses.append(valid_loss_per_epoch)
    
    return train_loss_per_epoch,valid_loss_per_epoch
    

def save_checkpoint(state, filename="Resnet_1.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def check_accuracy(loader, model, device=DEVICE):
    dice_score1=0
    loop = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for batch_idx, (data,label) in enumerate(loop):
            data = data.to(device=DEVICE,dtype=torch.float)
            label = label.to(device=DEVICE,dtype=torch.float)
           
            p1=model(data)
            
            p1 = (p1 > 0.5).float()
            dice_score1 += (2 * (p1 * label).sum()) / (
                (p1 + label).sum() + 1e-8)
           
    print(f"Dice score: {dice_score1/len(loader)}")

    model.train()
    return dice_score1/len(loader)
    


loss_func = torch.nn.CrossEntropyLoss()

 
epoch_len = len(str(NUM_EPOCHS))
                      
def main():
    model.to(device=DEVICE,dtype=torch.float)
    loss_fn1 =loss_func
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999),lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_loss,valid_loss=train_fn(train_loader,val_loader, model, optimizer, loss_fn1,scaler)
        
        print_msg = (f'[{epoch:>{epoch_len}}/{NUM_EPOCHS:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        dice_score= check_accuracy(val_loader, model, device=DEVICE)
        avg_valid_DS.append(dice_score.detach().cpu().numpy())
        

if __name__ == "__main__":
    main()


    