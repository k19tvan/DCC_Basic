import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torch.utils.data.dataloader
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import time
import os
import copy
from tqdm import tqdm 
from functional import save_model
from pathlib import Path
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dir = Path("/home/enn/workspace/generative_ai/pytorch/DogCatClassification")

# Transformation -> Dataloader -> Train function -> Non-Freeze Parameters -> Freeze Parameters
# Forward -> Loss -> Backward

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

# Hyperparameters
def parse_args():
    parser = argparse.ArgumentParser(description='DogCatClassificationTraining')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=str, default='True')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--freeze_params', type=str, default='True')
    parser.add_argument('--scheduler', type=str, default='True')
    parser.add_argument('--step', type=int, default=7)
    parser.add_argument('--gamma', type=float, default=0.1)
    
    return parser.parse_args()

args = parse_args()
batch_size = args.batch_size
shuffle = args.shuffle == 'True'
num_workers = args.num_workers
num_epochs = args.num_epochs
lr = args.lr
freeze_params = args.freeze_params == 'True'
step = args.step
gamma = args.gamma
scheduler = args.scheduler == 'True'

# Dataloader
data_dir = 'data'   

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
data_loaders = {x: DataLoader(dataset=image_datasets[x], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers) for x in ['train', 'val']}

datasets_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Train function
def count_files(dir):
    return len(list(dir.glob('*')))

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    
    #Save best.pth
    path = dir/Path("run")
    times = count_files(path) + 1
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to Train 
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            with tqdm(total=datasets_sizes[phase], desc=f'{phase} Epoch {epoch+1}/{num_epochs}', ncols=100, unit='img') as pbar:
                for (inputs, labels) in data_loaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        # Feedforward
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        
                        # Loss
                        loss = criterion(outputs, labels)
                        
                        if phase == 'train':
                            # Backward
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        
                        running_loss += loss.item() * inputs.size(0)  # inputs.size(0) = number of images in current batch
                        running_corrects += torch.sum(preds == labels.data)

                        pbar.update(inputs.size(0)) 

            if phase == 'train' and scheduler:
                scheduler.step()
                
            epoch_loss = running_loss / datasets_sizes[phase]
            epoch_acc = running_corrects / datasets_sizes[phase]
            
            print(f'{phase} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                save_model(model, path, times)
            
        print()
                    
    print(f'Best Accuracy: {best_acc} in {time.time() - since} second')
    model.load_state_dict(best_model_wts)
    
    return model

if __name__ == '__main__':
    
    print("Hyperparameters", "-"*20)
    print(f"batch_size = {batch_size}")
    print(f"shuffle = {shuffle}")
    print(f"num_workers = {num_workers}")
    print(f"num_epochs = {num_epochs}")
    print(f"lr = {lr}")
    print(f"freeze_params = {freeze_params}")
    print(f"scheduler = {scheduler}")
    print(f"step = {step}")
    print(f"gamma = {gamma}")
    print()

    print("Training", "-"*20)
    model = models.resnet18(weights='IMAGENET1K_V1')

    if freeze_params:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # Every 7 Epochs, Learning_rate *= 0.1

    model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs)




