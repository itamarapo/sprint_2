MAX_NUM_LAUNCHERS = 50
import random
import numpy as np

def center_of_mass(points):
    # Convert list of points to a NumPy array for easy manipulation
    points = np.array(points)

    # Calculate the mean of the x and y coordinates
    C_x = np.mean(points[:, 0])  # Mean of x-coordinates
    C_y = np.mean(points[:, 1])  # Mean of y-coordinates

    # Return the center of mass as a tuple (C_x, C_y)
    return C_x, C_y


def list_equal(A, B, ):
    """A function that checks weather two 2D lists of floats are equal"""
    if len(A) != len(B):
        return False
    for i in range(len(A)):
        if len(A[i]) != len(B[i]):
            return False
        for j in range(len(A[i])):
            if A[i][j] != B[i][j]:
                return False
    return True
def distance(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def find_nearest_point(point, launchers):
    min_dist = distance(point, launchers[0])
    index = 0
    for i in range(len(launchers)):
        if distance(point,launchers[i]) < min_dist:
            min_dist = distance(point,launchers[i])
            index = i
    return index


def choose_launchers(points, max_num_launchers):
    launchers = random.sample(points, max_num_launchers)
    return launchers


def improve_choose_launchers(groups):
    new_launchers = []
    for group in groups:
        group_locations = []
        for rocket in group:
            group_locations.append(rocket[0])
        new_launch = center_of_mass(group_locations)
        new_launchers.append(new_launch)
    return new_launchers


def separate_to_groups(data, launchers):
    group = []
    for i in range(len(launchers)):
        group.append([])
    for rocket in data:
        group[find_nearest_point(rocket[0], launchers)].append(rocket)
    return group


def find_launchers(data):
    points = []
    for rocket in data:
        points.append(rocket[0])
    launchers_even = choose_launchers(points, MAX_NUM_LAUNCHERS)
    groups = separate_to_groups(data, launchers_even)
    launchers_odd = improve_choose_launchers(groups)
    while not list_equal(launchers_even, launchers_odd):
        groups = separate_to_groups(data, launchers_odd)
        launchers_even = improve_choose_launchers(groups)
        groups = separate_to_groups(data, launchers_even)
        launchers_odd = improve_choose_launchers(groups)
    return launchers_even, groups


def all_data_launchers(data):
    launchers, groups = find_launchers(data)
    data_launchers = []
    for i in range(len(launchers)):
        data_launchers.append([])
        data_launchers[i].append(launchers[i])
        data_launchers[i].append([])
        for j in range(len(groups[i])):
            data_launchers[i][1].append(groups[i][j][2])
    return data_launchers
