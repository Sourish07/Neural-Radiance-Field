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

    # Calculate the offsets for each pixel
    offset_x = 1 / (2 * width)
    offset_y = 1 / (2 * height)

    # Creating the grid of points directly
    i, j = torch.meshgrid(
        torch.linspace(-image_width / 2 + offset_x, image_width / 2 + offset_x, width, device=device),
        torch.linspace(image_height / 2 + offset_y, -image_height / 2 + offset_y, height, device=device),
        indexing='ij'
    )
    
    # Calculate and normalize direction vectors
    directions = torch.stack([(i / focal_length), (j / focal_length), -torch.ones_like(i)], dim=0).reshape(3, -1)
    directions /= torch.norm(directions, dim=0, keepdim=True)
    rays_d = cam_2_world[:3, :3] @ directions
    rays_d = rays_d.reshape(3, height, width)
    rays_o = torch.broadcast_to(cam_2_world[:3, -1].reshape(3, 1, 1), (3, height, width))

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


