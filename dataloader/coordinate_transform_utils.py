

import numpy as np



def get_new_idx(x, a = 0., b = 0.5, lam = 1.0):
    
    g = (2 * b - 2 * a) * x + 2 * a
    f = x + 0.5 * lam * x * (np.sin(g) - np.sin(2 * b))
    return f



def center_norm(coord):
    
    x_min, x_max = np.min(coord[:, 0]), np.max(coord[:, 0])
    y_min, y_max = np.min(coord[:, 1]), np.max(coord[:, 1])
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = np.max([x_range, y_range])
    
    coord[:, 0] = (coord[:, 0] - x_min + (max_range - x_range) / 2.0) / max_range
    coord[:, 1] = (coord[:, 1] - y_min + (max_range - y_range) / 2.0) / max_range
    return coord



def coordinate_transform_utils(coord, a_x = 0., b_x = 0.5, c_x = 1.0, a_y = 0., b_y = 0.5, c_y = 1.0):
    
    coord[:, 0] = get_new_idx(coord[:, 0] / 255.0, a_x, b_x, c_x)  
    coord[:, 1] = get_new_idx(coord[:, 1] / 255.0, a_y, b_y, c_y)
    coord = center_norm(coord)
    return coord
