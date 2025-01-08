from pathlib import Path
import shutil
import random
import torch

def train_val_split(dir, s_class, val_ratio=0.2):

    train_path = dir/Path(f'data/train/{s_class}')
    val_path = dir/Path(f'data/val/{s_class}')
    
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    src = dir/Path('train/train')
    files = [file.relative_to(src) for file in list(src.glob(f"{s_class}*"))]

    random.shuffle(files)

    split_index = int(len(files) * (1 - val_ratio))
    for i in range(len(files)):
        print(files[i])
        file_name = Path(src/Path(files[i]))
        file_name.rename(train_path/files[i] if i < split_index else val_path/files[i])        
        
def count_files(dir):
    return len(list(dir.glob('*')))

def save_model(model, path, times):
    
    run_path = path/Path(f"run{times}")
    
    run_path.mkdir(parents=True, exist_ok=True)
    weight_path = run_path/Path("best.pth")
    torch.save(model, weight_path)
    
    
    

