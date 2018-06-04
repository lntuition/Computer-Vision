import numpy as np
import math

def transform(p, s_x, s_y, theta, t_x, t_y):
    p_s = np.dot(np.array([[s_x, 0, 0], [0, s_y, 0], [0, 0, 1]]), p)
    p_r = np.dot(np.array([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]]), p_s)
    p_t = np.dot(np.array([[1, 0, t_x], [0, 1, t_y], [0, 0, 1]]), p_r)

    return p_s, p_r, p_t
