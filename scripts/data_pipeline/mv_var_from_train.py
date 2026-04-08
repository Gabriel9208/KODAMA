import yaml
import shutil
from pathlib import Path

def mv_val():
    '''
    move the files that are in validation set under data/images/train to data/images/val and data/labels/train to data/labels/val
    the trainset and valset is under configs/split_config.yaml
    '''
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / 'configs' / 'split_config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    val_sessions = set(config['split']['val_sessions'])
    
    dirs_to_process = [
        (project_root / 'data' / 'images' / 'train', project_root / 'data' / 'images' / 'val'),
        (project_root / 'data' / 'labels' / 'train', project_root / 'data' / 'labels' / 'val')
    ]
    
    for train_dir, val_dir in dirs_to_process:
        if not train_dir.exists():
            continue
            
        val_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path in train_dir.iterdir():
            if not file_path.is_file():
                continue
                
            if any(session in file_path.name for session in val_sessions):
                dest_path = val_dir / file_path.name
                shutil.move(str(file_path), str(dest_path))
                print(f"Moved {file_path.name} to {val_dir}")

if __name__ == "__main__":
    mv_val()