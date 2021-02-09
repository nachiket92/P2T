from datasets.ns import NS
from torch.utils.data import DataLoader
import yaml

"""
Script to extract map images for nuscenes. This significantly speeds up training compared to using the predict helper
during training. 
"""

config_file = 'configs/ns.yml'
# Read config file
with open(config_file, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

# Initialize datasets and dataloaders:
batch_size = 64
tr_set = NS(config['dataroot'],
            config['train'],
            img_size=config['img_size'],
            grid_extent=config['grid_extent'],
            image_extraction_mode=True)

val_set = NS(config['dataroot'],
             config['val'],
             img_size=config['img_size'],
             grid_extent=config['grid_extent'],
             image_extraction_mode=True)

ts_set = NS(config['dataroot'],
            config['test'],
            img_size=config['img_size'],
            grid_extent=config['grid_extent'],
            image_extraction_mode=True)

tr_dl = DataLoader(tr_set, batch_size=batch_size, shuffle=False, num_workers=8)
val_dl = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
ts_dl = DataLoader(ts_set, batch_size=batch_size, shuffle=False, num_workers=8)

print('Extracting train set images....')
for k, data in enumerate(tr_dl):
    print('Batch ' + str(k) + ' of ' + str(len(tr_set) // batch_size + 1))

print('Extracting val set images....')
for k, data in enumerate(val_dl):
    print('Batch ' + str(k) + ' of ' + str(len(val_set) // batch_size + 1))

print('Extracting test set images....')
for k, data in enumerate(ts_dl):
    print('Batch ' + str(k) + ' of ' + str(len(ts_set) // batch_size + 1))
