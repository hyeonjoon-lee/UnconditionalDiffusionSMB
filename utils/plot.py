from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL

filepath = Path(__file__).parent.resolve()
parentpath = Path(__file__).parent.parent
DATA_PATH = os.path.join(parentpath, "levels", "ground")

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
}

sprites = {
    encoding["X"]: os.path.join(filepath, "sprites", "stone.png"),
    encoding["S"]: os.path.join(filepath, "sprites", "breakable_stone.png"),
    encoding["?"]: os.path.join(filepath, "sprites", "question.png"),
    encoding["Q"]: os.path.join(filepath, "sprites", "depleted_question.png"),
    encoding["E"]: os.path.join(filepath, "sprites", "goomba.png"),
    encoding["<"]: os.path.join(filepath, "sprites", "left_pipe_head.png"),
    encoding[">"]: os.path.join(filepath, "sprites", "right_pipe_head.png"),
    encoding["["]: os.path.join(filepath, "sprites", "left_pipe.png"),
    encoding["]"]: os.path.join(filepath, "sprites", "right_pipe.png"),
    encoding["o"]: os.path.join(filepath, "sprites", "coin.png"),
}

def get_img_from_level(level: np.ndarray) -> np.ndarray:
    """
    Given a level as a 2D numpy array of integers, return an image as a 3D numpy array.
    Each integer value in the input array is mapped to a corresponding sprite image, and the sprites are 
    arranged in a grid to form the output image.
    """
    image = []
    for row in level:
        image_row = []
        for c in row:
            if c == encoding["-"]:
                # Create a white tile for empty spaces
                tile = (255 * np.ones((16, 16, 3))).astype(int)
            elif c == -1:
                # Create a gray tile for masked spaces
                tile = (128 * np.ones((16, 16, 3))).astype(int)
            else:
                # Load the corresponding sprite image from disk and convert it to a numpy array
                tile = np.asarray(PIL.Image.open(sprites[c]).convert("RGB")).astype(int)
            image_row.append(tile)
        image.append(image_row)

    # Concatenate the tiles for each row horizontally to create a row-wise image
    image = [np.hstack([tile for tile in row]) for row in image]

    # Concatenate the row-wise images vertically to create a single output image
    image = np.vstack([np.asarray(row) for row in image])

    return image

def save_level_from_array(path: str, level: np.ndarray, title: str = None, dpi: int = 150) -> None:
    """
    Saves a level array as an image file at the given path.

    Args:
        path (str): The path to save the image file.
        level (np.ndarray): A 2D numpy array representing the level to plot.
        title (str, optional): The title of the plot. Defaults to None.
        dpi (int, optional): The resolution of the plot. Defaults to 150.
    """
    # Get the image of the level
    image = get_img_from_level(level)

    # Plot the image
    plt.imshow(255 * np.ones_like(image))  # White background
    plt.imshow(image)
    plt.axis("off")

    # Set the title if provided
    if title is not None:
        plt.title(title)

    # Save the plot to the given path
    plt.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close()


def plot_level_from_array(ax: plt.Axes, level: np.ndarray, title: str = None) -> None:
    """
    Plots a level array in the given Axes.

    Args:
        ax (plt.Axes): The Axes in which to plot the level.
        level (np.ndarray): A 2D numpy array representing the level to plot.
        title (str, optional): The title of the plot. Defaults to None.
    """
    # Get the image of the level
    image = get_img_from_level(level)

    # Plot the image in the given Axes
    ax.imshow(255 * np.ones_like(image))  # White background
    ax.imshow(image)
    ax.axis("off")

    # Set the title if provided
    if title is not None:
        ax.set_title(title)

if __name__ == '__main__':
    with np.load(os.path.join(DATA_PATH, 'unique_onehot.npz')) as data:
        levels = data['levels']
        for i in reversed(range(1, 10)):
            level = np.argmax(levels[-i], axis=0)
            save_level_from_array(level=level, path='img{:02d}'.format(10-i))