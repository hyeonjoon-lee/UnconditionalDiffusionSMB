import os
import torch as t
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from utils.plot import get_img_from_level

filepath = Path(__file__).parent.resolve()
DATA_PATH = os.path.join(filepath, "levels", "ground", "unique_onehot.npz")

class MarioDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        x = t.tensor(sample, dtype=t.float32)
        return x

def load_data(file_path):
    data = np.load(file_path)
    levels = data['levels']
    return levels

def create_dataloader(file_path, batch_size=32, shuffle=True, num_workers=0):
    data = load_data(file_path)
    dataset = MarioDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

if __name__ == '__main__':
    # mario_dataloader = create_dataloader(DATA_PATH, batch_size=64, shuffle=True)
    # # collect the first batch from mario_dataloader
    # first_batch = next((iter(mario_dataloader)))
    # print(first_batch.shape)
    # # plot all the levels in the first batch
    # fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    # for i, ax in enumerate(axes.flatten()):
    #     level = np.argmax(first_batch[i], axis=0).numpy()
    #     image = get_img_from_level(level)
    #     ax.imshow(255 * np.ones_like(image))  # White background
    #     ax.imshow(image)
    #     ax.axis("off")
    # plt.show()
    data = create_dataloader(DATA_PATH, batch_size=64, shuffle=True).dataset.data
    data = np.argmax(data, axis=1)
    # count the occurrence of each number in data
    for i in range(11):
        print(f"{i}: {np.count_nonzero(data == i)}")