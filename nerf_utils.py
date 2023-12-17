import numpy as np
import json
import torch
from PIL import Image


def read_in_dataset(idx=None, split='train'):
    assert split in ['train', 'val', 'test']
    with open(f'lego/transforms_{split}.json', "r") as f:
        data = json.load(f)



    if idx is None:
        return np.array([data['frames'][i]['transform_matrix'] for i in range(len(data['frames']))])
    else:
        return np.array(data['frames'][idx]['transform_matrix'])


def read_in_image(idx=0, split='train'):
    img = Image.open(f'lego/{split}/r_{idx}.png')

    # convert img to tensor
    img = np.array(img)

    # set all transparent pixels to white
    # img[..., 3] == 0 is the alpha channel
    # We're checking if it's transparent, i.e. if it's 0
    # the == 0 returns a boolean array, which we use to index into the image
    img[img[..., 3] == 0, :3] = 255

    return torch.tensor(img, device="cuda", dtype=torch.float32)
    

def get_rays(cam_2_world, width=800, height=800):
    device = cam_2_world.device
    
    aspect_ratio = width / height
    focal_length = 3
    image_height = 2
    image_width = aspect_ratio * image_height

    # Creating the grid of points
    i, j = torch.meshgrid(
        # top_left[0] is x coordinate of top left corner (left bound)
        # bottom_right[0] is x coordinate of bottom right corner (right bound)
        torch.linspace(-image_width / 2, image_width / 2, width, device=device) + (1 / (2 * width)),
        # top_left[1] is y coordinate of top left corner (top bound)
        # bottom_right[1] is y coordinate of bottom right corner (bottom bound)
        torch.linspace(image_height / 2, -image_height / 2, height, device=device) + (1 / (2 * height)),
        indexing='ij'
    )
    
    directions = torch.stack([i, j, -torch.ones_like(i, device=device) * focal_length])
    # normalize the directions
    directions /= torch.norm(directions, dim=0, keepdim=True)
    rays_d = cam_2_world[:3, :3] @ directions.reshape(3, -1)
    rays_d = rays_d.reshape(3, height, width)
    rays_o = torch.broadcast_to(cam_2_world[:3, -1], (height, width, 3)).permute(2, 0, 1)
    return rays_o, rays_d


def stratified_sampling(device="cuda"):
    # Stratified sampling
    N = 151
    t_near = 1
    t_far = 7

    u = torch.linspace(t_near, t_far, N + 1, device=device)[:-1]
    width_of_bin = (t_far - t_near) / N

    # generate N random samples from a uniform distribution
    t = torch.rand(N, dtype=torch.float32, device=device) * width_of_bin + u
    d = t[1:] - t[:-1]

    return t[1:], d


