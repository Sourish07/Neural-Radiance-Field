import numpy as np
import json
from PIL import Image


def read_in_data(idx=None, split='train'):
    assert split in ['train', 'val', 'test']
    with open(f'lego/transforms_{train}.json', "r") as f:
        data = json.load(f)

    if idx is None:
        return np.array([data['frames'][i]['transform_matrix'] for i in range(len(data['frames']))])
    else:
        return np.array(data['frames'][idx]['transform_matrix'])
    
def read_in_image(idx=0, split='train'):
    img = Image.open(f'lego/{split}/r_88.png')

    # convert img to tensor
    img = np.array(img)
    height, weight, _ = img.shape

    # set all transparent pixels to white
    # img[..., 3] == 0 is the alpha channel
    # We're checking if it's transparent, i.e. if it's 0
    # the == 0 returns a boolean array, which we use to index into the image
    img[img[..., 3] == 0, :3] = 255

    return img
    


def get_rays(width, height, cam_2_world):
    aspect_ratio = width / height
    focal_length = 1
    image_height = 2
    image_width = aspect_ratio * image_height

    # Creating the top left and bottom right corners of the image
    top_left = np.array([-image_width / 2, image_height / 2, -focal_length])
    bottom_right = np.array([image_width / 2, -image_height / 2, -focal_length])

    # Creating the grid of points
    i, j = np.meshgrid(
        # top_left[0] is x coordinate of top left corner (left bound)
        # bottom_right[0] is x coordinate of bottom right corner (right bound)
        np.linspace(top_left[0], bottom_right[0], width) + (1 / (2 * width)),
        # top_left[1] is y coordinate of top left corner (top bound)
        # bottom_right[1] is y coordinate of bottom right corner (bottom bound)
        np.linspace(top_left[1], bottom_right[1], height) + (1 / (2 * height)),
        indexing='ij'
    )
    i, j = i.T, j.T
    directions = np.stack([i, j, -np.ones_like(i)]).reshape((3, -1))

    rays_d = cam_2_world[:3, :3] @ directions
    rays_o = np.broadcast_to(cam_2_world[:3, [-1]], rays_d.shape)

    return rays_o, rays_d

def stratified_sampling():
    # Stratified sampling
    N = 192
    t_near = 0
    t_far = 6

    u = np.linspace(t_near, t_far, N + 1)[:-1]
    width_of_bin = (t_far - t_near) / N

    # generate N random samples from a uniform distribution
    t = np.rand(N) * width_of_bin + u
    d = t[1:] - t[:-1]

    return d


