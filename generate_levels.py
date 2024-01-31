import os,sys
import torch
from diffusion import Diffusion
from modules import UNet
import numpy as np
from matplotlib import pyplot as plt
from utils.plot import get_img_from_level
import argparse
from pathlib import Path
import math

FILE_PATH = Path(__file__).parent.resolve()
MODEL_PATH = os.path.join(FILE_PATH, "models")

def generate_levels(args):
    # if model path does not exist, assert error
    if not os.path.exists(os.path.join(MODEL_PATH, args.beta_schedule, args.temperature_scaling, f"ckpt_{args.epochs}.pt")):
        raise AssertionError("Model does not exist. Please train model first.")
    
    # if target path does not exist, create it
    if not os.path.exists(os.path.join(FILE_PATH, "generated_levels")):
        os.mkdir(os.path.join(FILE_PATH, "generated_levels"))

    trained_model = torch.load(os.path.join(MODEL_PATH, args.beta_schedule, args.temperature_scaling, f"ckpt_{args.epochs}.pt"))
    model = UNet().to(args.device)
    model.load_state_dict(trained_model['model_state_dict'])
    diffusion = Diffusion(img_size=14, device=args.device, schedule=args.beta_schedule)
    sampled_levels = diffusion.sample(model, n=args.num_levels)[-1].cpu().numpy()

    fig = plt.figure(figsize=(60, 30))
    num_cols = 2*int(math.sqrt(args.num_levels/2))
    num_rows = math.ceil(args.num_levels/num_cols)
    for i in range(args.num_levels):
        ax1 = fig.add_subplot(num_rows, num_cols, i+1)
        ax1.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        level = np.argmax(sampled_levels[i], axis=0)
        level_img = get_img_from_level(level)
        ax1.imshow(level_img)
    
    plt.savefig(os.path.join(FILE_PATH, "generated_levels", f"{args.beta_schedule}_{args.temperature_scaling}_{args.epochs}.png"))
    np.savez(os.path.join(FILE_PATH, "generated_levels", f"{args.beta_schedule}_{args.temperature_scaling}_{args.epochs}.npz"), levels=sampled_levels)

def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--beta_schedule", type=str, default="quadratic", choices=['linear', 'quadratic', 'sigmoid'])
    parser.add_argument("--temperature_scaling", type=str, default="fine", choices=['moderate', 'coarse', 'fine'])
    parser.add_argument("--num_levels", type=int, default=100)
    args = parser.parse_args()
    return generate_levels(args)

if __name__ == "__main__":
    launch()