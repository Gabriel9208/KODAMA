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

    shards = [
        v["shard_files"][0] 
        for k, v in progress["sessions"].items() if v["status"] == "cleaned"
    ]
    
    shards = list(set(shards))

    train_shards, val_shards = train_test_split(
        shards, 
        test_size=split_config["split"]["val_portion"],
        random_state=split_config["split"]["seed"]
    )

    split_config["split"]["train_shards"] = train_shards
    split_config["split"]["val_shards"] = val_shards

    with open('configs/split_config.yaml', 'w') as f:
        yaml.dump(split_config, f)

    

if __name__ == "__main__":
    split_dataset(config, 42)