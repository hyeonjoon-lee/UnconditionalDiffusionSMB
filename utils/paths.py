import subprocess
from typing import List
from itertools import product
import json
import numpy as np
import os, sys
import glob
from pathlib import Path
import multiprocessing
from multiprocessing import Pool

filepath = Path(__file__).parent.parent.parent
JARFILE_PATH = os.path.join(filepath, "simulator.jar")
DATA_PATH = os.path.join(filepath, "paths")

encoding = {
    "X": 0,
    "S": 1,
    "-": 2,
    "?": 3,
    "Q": 4,
    "E": 5,
    "<": 6,
    ">": 7,
    "[": 8,
    "]": 9,
    "o": 10,
    "B": 2,
    "b": 2,
    "x": 11
}

def level_to_arr(level_txt):
    txt_lev = level_txt.split("\n")
    str_lev = [list(row) for row in txt_lev if row != ""]

    # Convert characters in each row to integers using the encoding dictionary
    for row in str_lev:
        for i, c in enumerate(row):
            row[i] = encoding[c]

    # Create a sliding window of size 14x14 that covers the entire level
    window_lev = []
    for i in range(len(str_lev[0])-13):
        rows = []
        for j in range(14):
            rows.append(str_lev[j][i:i+14])
        window_lev.append(rows)

    # Convert the windowed level to a 3D NumPy array and return it
    return np.array(window_lev)

def levels_to_onehot(levels: np.ndarray, n_sprites: int=12):
    batch_size, h, w = levels.shape
    onehot = np.zeros((batch_size, n_sprites, h, w))

    # Iterate over each level in the input array
    for batch, level in enumerate(levels):
        # Iterate over each position in the level
        for i, j in product(range(h), range(w)):
            # Get the integer value of the sprite at the current position
            sprite = int(level[i][j])
            # Set the corresponding element of the one-hot encoded array to 1
            onehot[batch, sprite, i, j] = 1

    return onehot

def txt_to_onehot(path: str, unique: bool=False):
    files = glob.glob(os.path.join(path, '*.txt'))
    levels = np.empty((0, 14, 14), dtype=int)

    # Preprocess levels using multiple processes
    with Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(level_to_arr, [open(file, 'r').read() for file in files])

    # Fill in the levels array with the preprocessed levels
    for _, level in enumerate(results):
        levels = np.concatenate((levels, level), axis=0)
    print("Before unique: ", levels.shape)
    if unique:
        levels = np.unique(levels, return_index=False, axis=0)

    print("After unique: ", levels.shape)

    # Update the path in the levels array to reflect diagonal movement
    for i in range(levels.shape[0]):
        for j in range(levels.shape[2]):
            column = levels[i, :, j]
            ind = np.where(column==11)[0] if np.where(column==11)[0].size > 1 else None
            if ind is not None:
                if j == levels.shape[2]-1:
                    column[column==11] = 2
                    column[ind[-1]] = 11
                    levels[i, :, j] = column
                else:
                    if levels[i, ind[0], j+1] == 11:
                        column[ind[0]] = 2
                        levels[i, :, j] = column

    # Count the number of levels that have less than ten 11s (x)
    print("Number of levels with less than five 11s: ", np.count_nonzero(np.count_nonzero(levels==11, axis=(1, 2))<10))

    # Remove levels that have less than ten 11s (x)
    levels = levels[np.count_nonzero(levels==11, axis=(1, 2)) >= 10]
    print("After removing levels with less than five 11s: ", levels.shape)

    # Shuffle the levels
    np.random.shuffle(levels)

    # Convert levels to one-hot encoded arrays
    onehot = levels_to_onehot(levels)

    # Extract only 11s (x) from levels and set data type to int
    paths = onehot[:, 11, :, :].astype(int)

    levels = np.argmax(onehot, axis=1)
    # Change all 11s (x) to 2s (-)
    levels[levels==11] = 2
    onehot_levels = levels_to_onehot(levels, n_sprites=11)

    print("Paths shape: ", paths.shape)
    print("Onehot Levels shape: ", onehot_levels.shape)

    # Divide the levels and paths into training and validation sets
    train_size = int(0.8 * onehot_levels.shape[0])
    train_levels = onehot_levels[:train_size]
    train_paths = paths[:train_size]
    val_levels = onehot_levels[train_size:]
    val_paths = paths[train_size:]

    # Save one-hot encoded arrays as .npy file
    np.save(os.path.join(DATA_PATH, 'train_maps'), train_levels)
    np.save(os.path.join(DATA_PATH, 'train_shortest_paths'), train_paths)
    np.save(os.path.join(DATA_PATH, 'val_maps'), val_levels)
    np.save(os.path.join(DATA_PATH, 'val_shortest_paths'), val_paths)

def run_level(
    level: str,
    human_player: bool = False,
    max_time: int = 30,
    visualize: bool = False,
) -> dict:
    # Run the simulator.jar file with the given level
    if human_player:
        java = subprocess.Popen(
            ["java", "-cp", JARFILE_PATH, "geometry.PlayLevel", level],
            stdout=subprocess.PIPE,
        )
    else:
        java = subprocess.Popen(
            [
                "java",
                "-cp",
                JARFILE_PATH,
                "geometry.EvalLevel",
                level,
                str(max_time),
                str(visualize).lower(),
            ],
            stdout=subprocess.PIPE,
        )

    lines = java.stdout.readlines()
    res = lines[-1]
    res = json.loads(res.decode("utf8"))
    res["level"] = level

    return res

def test_level_from_int_array(
    level: np.ndarray,
    human_player: bool = False,
    max_time: int = 45,
    visualize: bool = True,
) -> dict:
    level = level.astype(int).tolist()
    level = str(level)

    return run_level(
        level, human_player=human_player, max_time=max_time, visualize=visualize
    )

if __name__ == '__main__':
    txt_to_onehot(path=DATA_PATH, unique=True)
    
    # # open npy file and display levels and paths
    # levels = np.load(os.path.join(DATA_PATH, 'unique_onehot.npy'))
    # paths = np.load(os.path.join(DATA_PATH, 'unique_paths.npy'))

    # for i in [2600, 2700, 2800]:
    #     print(np.argmax(levels[i], axis=0)) 
    #     print(paths[i])
    #     print()