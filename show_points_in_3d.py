import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

matplotlib.use('TkAgg')

from mpl_toolkits.mplot3d import Axes3D

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from scipy.integrate import solve_ivp
# from scipy.optimize import minimize
#
# def get_cartezian_coordinates(file_path:str, missleId:int):
#
#
#     # Read the CSV file
#     df = pd.read_csv(file_path)
#     df = df[df['ID'] == missleId]
#
#     # Assuming your CSV has columns named 'Range', 'Elevation', and 'Azimuth'
#     range_data = df['range'].to_numpy()
#     elevation_data = df['elevation'].to_numpy()
#     azimuth_data = df['azimuth'].to_numpy()
#     time_data = df['time'].to_numpy()
#
#     # Convert degrees to radians
#     elevation_rad = np.radians(elevation_data)
#     azimuth_rad = np.radians(azimuth_data)
#
#     # Convert spherical to Cartesian coordinates
#     x = np.array(range_data) * np.cos(elevation_rad) * np.cos(azimuth_rad)
#     y = np.array(range_data) * np.cos(elevation_rad) * np.sin(azimuth_rad)
#     z = np.array(range_data) * np.sin(elevation_rad)
#     coordinates = np.vstack((x, y, z)).T
#     return coordinates
#
#
#
#
# import numpy as np
# from scipy.optimize import curve_fit
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# import pandas as pd
# from scipy.interpolate import interp1d
#
# import show_points_in_3d
# from scipy.optimize import minimize
# matplotlib.use('TkAgg')
# import pyproj
# from scipy.integrate import solve_ivp
# from mpl_toolkits.mplot3d import Axes3D
#
# from mpl_toolkits.mplot3d import Axes3D
#
# # Define the parametric equations for a helix
# def helix_model(t, r, c, x0, y0, z0):
#     x = x0 + r * np.cos(t)
#     y = y0 + r * np.sin(t)
#     z = z0 + c * t
#     return np.concatenate((x, y, z))  # Concatenate x, y, z into a single array
#
# # Generate synthetic data for a helix
# np.random.seed(0)
# t_data = np.linspace(0, 4 * np.pi, 100)  # Parameter t
# r_true = 1.0  # True radius
# c_true = 0.2  # True rise per unit t
# x0_true, y0_true, z0_true = 0, 0, 0  # Starting point
# combined_data = helix_model(t_data, r_true, c_true, x0_true, y0_true, z0_true)
# # print(combined_data)
# # combined_data = get_cartezian_coordinates('With ID/Impact points data/Ashdod_with_ID.csv', 19)
# # print("combined_data")
# # print(combined_data)
#
#
# # Split the concatenated data into x_data, y_data, and z_data
# x_data, y_data, z_data = np.split(combined_data, 3)
#
# # Add some noise to the data
# x_data += np.random.normal(0, 0.1, len(t_data))
# y_data += np.random.normal(0, 0.1, len(t_data))
# z_data += np.random.normal(0, 0.1, len(t_data))
#
#
# # Fit the model to the data using curve_fit
# initial_guess = [8.0, 0.2, 0, 0, 0]  # Initial guess for r, c, x0, y0, z0
# params_opt, _ = curve_fit(lambda t, r, c, x0, y0, z0: helix_model(t, r, c, x0, y0, z0),
#                           t_data, np.concatenate((x_data, y_data, z_data)), p0=initial_guess)
#
# # Extract the fitted parameters
# r_fit, c_fit, x0_fit, y0_fit, z0_fit = params_opt
# print(f"Fitted parameters: r = {r_fit}, c = {c_fit}, x0 = {x0_fit}, y0 = {y0_fit}, z0 = {z0_fit}")
#
# # Plot the original data and the fitted helix
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x_data, y_data, z_data, color='b', label='Data')
#
# # Generate the fitted helix
# t_fit = np.linspace(0, 4 * np.pi, 100)
# combined_fit = helix_model(t_fit, *params_opt)
#
# # Split the fitted data into x_fit, y_fit, z_fit
# x_fit, y_fit, z_fit = np.split(combined_fit, 3)
#
# ax.plot(x_fit, y_fit, z_fit, color='r', label='Fitted Helix')
#
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.legend()
#
# plt.show()