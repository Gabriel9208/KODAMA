import yaml


def split_dataset(config: dict) -> None:
    with open('configs\split_config.yaml', 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    if data == None:
        print("No config file found.")
        return


    

if __name__ == "__main__":
    split_dataset(config)