import yaml



config = {
    "train" : "./dataset/images/train",
    "val" : "./dataset/images/val",
    "test": "./dataset/images/test",
    "nc" : 3,
    "names" : ['helmet', 'head', 'person']
}

with open('data.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)