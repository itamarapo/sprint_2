import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data from CSV
file_path = 'impactdata/Carmel_with_IDimpact.csv'  # Replace with your CSV file path

# Read the CSV file
df = pd.read_csv(file_path)

num_rockets = np.maximum(df['ID'].to_numpy().array())

rocket_dict = dict()
for i in range(1, num_rockets+1):
    rocket_i_data = df[df['ID'] == i]

    range_data = df['range'].to_numpy()
    elevation_data = df['elevation'].to_numpy()
    azimuth_data = df['azimuth'].to_numpy()
    radial_velocity_data = df['radial_velocity'].to_numpy()
    time_data = df['time'].to_numpy()

    # Convert degrees to radians
    elevation_rad = np.radians(elevation_data)
    azimuth_rad = np.radians(azimuth_data)

    # Convert spherical to Cartesian coordinates
    x = np.array(range_data) * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = np.array(range_data) * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = np.array(range_data) * np.sin(elevation_rad)
    t = np.array(time_data)

    # Plotting the points in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(x, y, z, c='b', marker='o')
    ax.scatter(x[0], y[0], z[0], c='r', marker='o')

    # Label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show plot
    plt.show()









