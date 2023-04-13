import subprocess
from typing import List
from itertools import product
import json
import numpy as np
import os
import glob
from pathlib import Path
import multiprocessing
from multiprocessing import Pool

filepath = Path(__file__).parent.parent
JARFILE_PATH = os.path.join(filepath, "simulator.jar")
DATA_PATH = os.path.join(filepath, "levels", "ground")

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
    "b": 2
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

def levels_to_onehot(levels: np.ndarray, n_sprites: int=11):
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

def txt_to_onehot(path: str, target: str, unique: bool=False):
    files = glob.glob(os.path.join(path, '*.txt'))
    levels = np.empty((0, 14, 14), dtype=int)

    # Preprocess levels using multiple processes
    with Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(level_to_arr, [open(file, 'r').read() for file in files])

    # Fill in the levels array with the preprocessed levels
    for _, level in enumerate(results):
        levels = np.concatenate((levels, level), axis=0)

    if unique:
        u, ind = np.unique(levels, return_index=True, axis=0)
        levels = u[np.argsort(ind)]
        target = 'unique_' + target

    # Convert levels to one-hot encoded arrays
    onehot = levels_to_onehot(levels)

    # Save one-hot encoded arrays as .npz file
    np.savez(os.path.join(DATA_PATH, target), levels=onehot)

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
    # txt_to_onehot(path=DATA_PATH, target='onehot', unique=True)

    with np.load(os.path.join(DATA_PATH, 'onehot.npz')) as data:
        levels = data['levels']
        levels = np.argmax(levels, axis=1)
        unique = np.unique(levels, axis=0)

        print('all levels: {}'.format(levels.shape))
        print('unique levels: {}\n'.format(unique.shape))
        
        for i in range(11):
            print('>>> no. of {}'.format(i))
            print('in all levels: \t\t{}'.format(np.count_nonzero(levels==i)))
            print('in unique levels: \t{}\n'.format(np.count_nonzero(unique==i)))

        test_level_from_int_array(levels[-1])