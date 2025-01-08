from pathlib import Path
import random
from functional import train_val_split

dir = Path("/home/enn/workspace/generative_ai/pytorch/DogCatClassification/")
     
train_val_split(dir, "dog")
train_val_split(dir, "cat")


