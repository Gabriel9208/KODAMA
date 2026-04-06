import yaml
import json
from sklearn.model_selection import train_test_split

def check_session() -> bool:
    """
        Check if sessions are assigned to the exactly one shard.
    """
    
    with open('data/pipeline_progress.json') as f:
        progress = json.load(f)

    split_sessions = progress.get("sessions", {}).get("train", progress.get("sessions", {}))

    overlap = False
    for sid, v in split_sessions.items():
        if "shard_files" not in v:
            continue
        if len(v["shard_files"]) > 1:
            overlap = True
            print(f"⚠️ Session {sid} spans multiple shards: {v['shard_files']}")

    if overlap:
        print("Overlap detected. Please run the pipeline again to fix the issue.")
        return
    
    print("No overlap detected.")

    return overlap
    

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

    split_sessions = progress.get("sessions", {}).get("train", progress.get("sessions", {}))

    sessions = [
        sid
        for sid, v in split_sessions.items() if v["status"] == "cleaned"
    ]

    train_sessions, val_sessions = train_test_split(
        sessions,
        test_size=split_config["split"]["val_portion"],
        random_state=split_config["split"]["seed"]
    )

    split_config["split"]["train_sessions"] = sorted(list(train_sessions))
    split_config["split"]["val_sessions"] = sorted(list(val_sessions))

    with open('configs/split_config.yaml', 'w') as f:
        yaml.dump(split_config, f)

    

if __name__ == "__main__":
    if not check_session():
        split_dataset()