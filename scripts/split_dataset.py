import yaml
import json
from sklearn.model_selection import train_test_split


def split_dataset() -> None:
    with open('data/pipeline_progress.json') as f:
        progress = json.load(f)

    with open('configs/split_config.yaml') as f:
        split_config = yaml.safe_load(f)
    
    if progress is None:
        print("No data/pipeline_progress.json file found.")
        return

    if split_config is None:
        print("No configs/split_config.yaml file found.")
        return

    sessions = [
        sid 
        for sid, v in progress["sessions"].items() if v["status"] == "cleaned"
    ]
    
    for sid in sessions:
        if len(progress["sessions"][sid]["shard_files"]) > 1:
            print(f"⚠️ Session {sid} spans multiple shards: {progress['sessions'][sid]['shard_files']}")
    
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
    conflict_shards = val_shards - val_shards_clean
    if conflict_shards:
        print(f"{len(conflict_shards)} shards contain both train and val sessions.")
        print(f"   These shards are assigned to train. Val samples inside are excluded.")
        print(f"   Conflict shards: {conflict_shards}")
        print(f"   Remain portion of validation shards: {len(val_shards_clean) / len(sessions):.3f}")

    split_config["split"]["train_shards"] = sorted(list(train_shards))
    split_config["split"]["val_shards"] = sorted(list(val_shards_clean))

    with open('configs/split_config.yaml', 'w') as f:
        yaml.dump(split_config, f)

    

if __name__ == "__main__":
    split_dataset()