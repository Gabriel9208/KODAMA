import yaml
import json
from sklearn.model_selection import train_test_split


def split_dataset(config: dict, seed: int) -> None:
    with open('data/pipeline_progress.json') as f:
        progress = json.load(f)

    with open('configs/split_config.yaml') as f:
        split_config = yaml.safe_load(f)
    
    if progress == None:
        print("No data/pipeline_progress.json file found.")
        return

    if split_config == None:
        print("No configs/split_config.yaml file found.")
        return

    sessions = [
        sid 
        for sid, v in progress["sessions"].items() if v["status"] == "cleaned"
    ]
    
    train_sessions, val_sessions = train_test_split(
        sessions, 
        test_size=split_config["split"]["val_portion"],
        random_state=split_config["split"]["seed"]
    )

    train_shards = set([
        shard_files
        for sid in train_sessions
        for shard_files in progress["sessions"][sid]["shard_files"]
    ])

    val_shards = set([
        shard_files
        for sid in val_sessions
        for shard_files in progress["sessions"][sid]["shard_files"]
    ])

    val_shards_clean = val_shards - train_shards
     

    split_config["split"]["train_shards"] = list(train_shards)
    split_config["split"]["val_shards"] = list(val_shards_clean)

    with open('configs/split_config.yaml', 'w') as f:
        yaml.dump(split_config, f)

    

if __name__ == "__main__":
    split_dataset(config, 42)