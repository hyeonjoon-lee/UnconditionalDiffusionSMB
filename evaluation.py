import numpy as np
import os
from pathlib import Path
import Levenshtein
from utils.plot import get_img_from_level
from utils.levels import test_level_from_int_array

moderate_quadratic = os.path.join(Path(__file__).parent.resolve(), "levels_moderate_quadratic.npz")
coarse_quadratic = os.path.join(Path(__file__).parent.resolve(), "levels_coarse_quadratic.npz")
mariogpt = os.path.join(Path(__file__).parent.resolve(), "levels_mariogpt.npz")

# Assume levels is an array of levels with shape (num_levels, 11, 14, 14)
# Example:
# levels = np.random.randint(0, 2, (100, 11, 14, 14))

def level_to_str(level):
    level_2d = level.argmax(axis=0)
    return ''.join([''.join(map(str, row)) for row in level_2d])

def compute_edit_distance(levels):
    assert levels.shape[1:] == (11, 14, 14), "Input levels must have shape (num_levels, 11, 14, 14)"
    distances = []
    n = len(levels)
    for i in range(n):
        level_yours_str = level_to_str(levels[i])
        
        for j in range(i + 1, n):
            level_another_str = level_to_str(levels[j])
            distance = Levenshtein.distance(level_yours_str, level_another_str)
            distances.append(distance)
            
    return np.mean(distances)

def compute_coverage(levels, threshold):
    assert levels.shape[1:] == (11, 14, 14), "Input levels must have shape (num_levels, 11, 14, 14)"
    count = 0
    n = len(levels)
    for i in range(n):
        level_yours_str = level_to_str(levels[i])
        
        for j in range(i + 1, n):
            level_another_str = level_to_str(levels[j])
            distance = Levenshtein.distance(level_yours_str, level_another_str)
            
            if distance <= threshold:
                count += 1
                break
                
    return count / n

if __name__ == '__main__':
    paths = [moderate_quadratic, coarse_quadratic, mariogpt]
    models = ["Moderate Quadratic", "Coarse Quadratic", "MarioGPT"]
    for i, path in enumerate(paths):
        levels = np.load(path)['levels']
        edit_distance = compute_edit_distance(levels)
        coverage = compute_coverage(levels, threshold=10) # Set threshold based on the acceptable difference between levels
        unique = np.unique(np.argmax(levels, axis=1), axis=0)
        
        playable_levels = 0
        playability_levels = np.argmax(levels, axis=1)
        for j in range(100):
            res = test_level_from_int_array(playability_levels[j], max_time=25, visualize=True)
            if res['marioStatus']==1:
                playable_levels += 1

        print(f">>> {models[i]}: {levels.shape[0]} levels")
        print('unique levels: {}'.format(unique.shape[0]/levels.shape[0]))
        print("Edit Distance:", edit_distance)
        print("Coverage:", coverage)
        print("Playability:", playable_levels/100)