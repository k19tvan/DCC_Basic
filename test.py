import torch
import argparse
from torchvision import transforms
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='DogCatClassificationTraining')
    parser.add_argument('--model_path', type=str, default="weight.pth")
    parser.add_argument('--img_path', type=str, default="test/test/1.jpg")
    
    return parser.parse_args()

args = parse_args()
model_path = args.model_path
img_path = args.img_path

model = torch.load(model_path)
model.eval()

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

img = data_transform(Image.open(img_path)).unsqueeze(0)

with torch.no_grad():
    img = img.to(device)
    output = model(img)
    _, predicted = torch.max(output, 1)
    
labels = ["Cat", "Dog"]
print(f"It is predicted to be a {labels[predicted.item()]}")

