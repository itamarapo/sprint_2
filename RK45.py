import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy.interpolate import interp1d

from scipy.optimize import minimize
from final_ringer_from_fink import *

matplotlib.use('TkAgg')
import pyproj
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# known conditions
g = 9.81
rockets = {}
rockets['Qassam 4'] = {'mass': 20, 'friction': 2.888, 'initialVel': 1090, 'A': 0.01, 'color': 'blue'}
rockets['Grad'] = {'mass': 160, 'friction': 11.121, 'initialVel': 1240, 'A': 0.047, 'color': 'green'}
rockets['M75'] = {'mass': 400, 'friction': 0.77, 'initialVel': 1270, 'A': 0.031, 'color': 'red'}
rockets['R160'] = {'mass': 670, 'friction': 1.243, 'initialVel': 2640, 'A': 0.072, 'color': 'black'}

total_time_of_flight = 58
number_of_point = 20
# this is just for the start
mass = 1
friction = 0
initialVel = 0
rho = 1.204
A = 0


# Function for projectile motion with air resistance
def projectile_motion(t, y):
    x, y, z, vx, vy, vz = y
    v = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)  # speed of the projectile

    # Drag force magnitude
    drag_force = 1 / 70 * rho * A * friction * v ** 2
    drag_acceleration = drag_force / mass

    # Calculate acceleration components (including gravity)
    ax = -drag_acceleration * (vx / v)  # acceleration in x due to drag
    ay = -drag_acceleration * (vy / v)  # acceleration in y due to drag
    az = -g - drag_acceleration * (vz / v)  # acceleration in z due to gravity and drag

    return [vx, vy, vz, ax, ay, az]


# Function to predict the trajectory with given initial conditions
def predict_trajectory(initial_conditions, time=total_time_of_flight):
    # Unpack initial conditions
    x0, y0, z0, v0, angle_xy, angle_z = initial_conditions

    # Convert angles to radians and compute initial velocity components
    vx0 = v0 * np.cos(np.radians(angle_xy)) * np.cos(np.radians(angle_z))
    vy0 = v0 * np.sin(np.radians(angle_xy))
    vz0 = v0 * np.cos(np.radians(angle_xy)) * np.sin(np.radians(angle_z))

    # Initial state vector: [x, y, z, vx, vy, vz]
    y0 = [x0, y0, z0, vx0, vy0, vz0]

    # Time span for the simulation
    t_span = (-time, time * 4)  # Simulate for the specified total time
    t_eval = np.linspace(-time, time * 4, number_of_point * 30)  # Time points for evaluation

    # Solve the ODE using RK45 (Runge-Kutta method)
    solution = solve_ivp(projectile_motion, t_span, y0, method='RK45', t_eval=t_eval)
    x = solution.y[0]  # position in x
    y = solution.y[1]  # position in y
    z = solution.y[2]  # position in z
    vx = solution.y[3]  # velocity in x
    vy = solution.y[4]  # velocity in y
    vz = solution.y[5]  # velocity in z

    return solution.t, x, y, z  # Return time, x, y, z positions


# Objective function to minimize (sum of squared errors between predicted and given coordinates)
def error_function(initial_conditions, given_coords, total_time=total_time_of_flight):
    # Predict the trajectory based on the current initial conditions

    t_eval, predicted_x, predicted_y, predicted_z = predict_trajectory(initial_conditions, total_time)
    # Interpolate the predicted trajectory to match the given coordinates' number of points
    given_t = np.linspace(0, total_time_of_flight,
                          len(given_coords['x']))  # Assume given coords are evenly spaced over the total time

    # Interpolate predicted trajectory to match the given time points
    interp_x = interp1d(t_eval, predicted_x, kind='linear', fill_value="extrapolate")
    interp_y = interp1d(t_eval, predicted_y, kind='linear', fill_value="extrapolate")
    interp_z = interp1d(t_eval, predicted_z, kind='linear', fill_value="extrapolate")

    # Interpolate the predicted coordinates
    interp_pred_x = interp_x(given_t)
    interp_pred_y = interp_y(given_t)
    interp_pred_z = interp_z(given_t)

    # Calculate the error (sum of squared differences between predicted and actual coordinates)
    error = np.sum((interp_pred_x - given_coords['x'].to_numpy()) ** 2 +
                   (interp_pred_y - given_coords['y'].to_numpy()) ** 2 +
                   (interp_pred_z - given_coords['z'].to_numpy()) ** 2)

    return error


# Main function to find the initial conditions that best fit the given coordinates
def fit_trajectory_to_coordinates(given_coords, initial_guess, total_time=total_time_of_flight):
    # Use a minimizer to find the best initial conditions that minimize the error
    result = minimize(error_function, initial_guess, args=(given_coords, total_time), method='Nelder-Mead')

    # Get the optimized initial conditions
    optimized_initial_conditions = result.x

    # Predict the trajectory with the optimized initial conditions
    t_eval, optimized_x, optimized_y, optimized_z = predict_trajectory(optimized_initial_conditions, total_time)

    return t_eval, optimized_x, optimized_y, optimized_z, optimized_initial_conditions


def get_cartezian_coordinates(file_path: str, missleId: int):
    # Read the CSV file
    df = pd.read_csv(file_path)
    df = df[df['ID'] == missleId]

    # Assuming your CSV has columns named 'Range', 'Elevation', and 'Azimuth'
    range_data = df['range'].to_numpy()
    elevation_data = df['elevation'].to_numpy()
    azimuth_data = df['azimuth'].to_numpy()
    time_data = df['time'].to_numpy()

    # Convert degrees to radians
    elevation_rad = np.radians(elevation_data)
    azimuth_rad = np.radians(azimuth_data)

    # Convert spherical to Cartesian coordinates
    x = np.array(range_data) * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = np.array(range_data) * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = np.array(range_data) * np.sin(elevation_rad)
    coordinates = np.vstack((x, y, z, time_data)).T
    return coordinates


# Example usage
if __name__ == "__main__":
    # Example given coordinates (these would typically be taken from your dataset)
    # The coordinates should be in the form: [(x1, y1, z1, t), (x2, y2, z2, t), ...]

    data = get_list_of_df(True)
    print(type(data))
    allOfTheFuckingCoordinets=[]
    x_lunch=[]
    y_lunch=[]
    x_target=[]
    y_target=[]
    for given_coords in data[:90]:
        # l=len(given_coords[x'])
        # given_coords['x']=given_coords['x'][int(l*2/3):l]
        # given_coords['y']=given_coords['y'][int(l*2/3):l]
        # given_coords['z']=given_coords['z'][int(l*2/3):l


        answers = {}
        for rocket in rockets:
            friction = rockets[rocket]['friction']
            mass = rockets[rocket]['mass']
            initialVel = rockets[rocket]['initialVel']
            A = rockets[rocket]['A']
            # Initial guess for the optimization: [x0, y0, z0, v0, angle_xy, angle_z,t,m]
            initial_guess = [given_coords['x'][0], given_coords['y'][0], given_coords['z'][0], initialVel, 45,
                             30]  # initial guess
            # Fit the trajectory to the given coordinates
            t_eval, optimized_x, optimized_y, optimized_z, optimized_initial_conditions = fit_trajectory_to_coordinates(
                given_coords, initial_guess)
            er = error_function(optimized_initial_conditions, given_coords)

            answers[rocket] = [er, t_eval, optimized_x, optimized_y, optimized_z, optimized_initial_conditions]

        minRocket = 'Qassam 4'
        minError = answers[minRocket][0]
        for rocket in answers:
            if answers[rocket][0] < minError:
                minError = answers[rocket][0]
                minRocket = rocket
        # print(f"minRocket: {minRocket}")


        er, t_eval, optimized_x, optimized_y, optimized_z, optimized_initial_conditions = answers[minRocket]

        lunchIndex = 0
        hitIndex = 0
        i = 0
        while (i < len(optimized_z) - 1):
            if optimized_z[i] < 0 and 0 < optimized_z[i + 1]:
                lunchIndex = i
            if optimized_z[i] > 0 and 0 > optimized_z[i + 1]:
                hitIndex = i
            i = i + 1
        lunchCoor = [optimized_x[lunchIndex] + optimized_x[lunchIndex + 1] / 2,
                     optimized_y[lunchIndex] + optimized_y[lunchIndex + 1] / 2]
        hitCoor = [optimized_x[hitIndex] + optimized_x[hitIndex + 1] / 2,
                   optimized_y[hitIndex] + optimized_y[hitIndex + 1] / 2]
        x_,y_,z_ = convert_from_ashdod(lunchCoor[0], lunchCoor[1], 0)
        lat, log = xyz_to_lat_and_lon(x_,y_,z_)
        lunchCoor[0]=lat
        lunchCoor[1]=log

        x_lunch.append(lat)
        y_lunch.append(log)

        x_, y_, z_ = convert_from_ashdod(hitCoor[0], hitCoor[1], 0)
        lat, log = xyz_to_lat_and_lon(x_, y_, z_)
        hitCoor[0] = lat
        hitCoor[1] = log


        x_target.append(lat)
        y_target.append(log)
        allOfTheFuckingCoordinets.append([lunchCoor, hitCoor, minRocket])
        print([lunchCoor, hitCoor, minRocket])
    print(allOfTheFuckingCoordinets)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    # for rocket in rockets:
    #     er, t_eval, optimized_x, optimized_y, optimized_z, optimized_initial_conditions = answers[rocket]
    #     ax.plot(optimized_x, optimized_y, optimized_z, label=f"Predicted Trajectory:{rocket}",
    #             color=rockets[rocket]['color'])

    # Plot the optimized trajectory
    # ax.plot(optimized_x, optimized_y, optimized_z, label="Predicted Trajectory", color='blue')

    # Plot the given points
    ax.scatter(x_lunch, y_lunch, color='red', label="Given Locations")

    ax.set_title("Optimized 3D Missile Trajectory with Drag")
    ax.set_xlabel("Horizontal Distance X (m)")
    ax.set_ylabel("Horizontal Distance Y (m)")
    # ax.set_zlabel("Vertical Distance Z (m)")
    ax.grid(True)
    ax.legend()

    plt.show()

    ax.scatter(x_target, y_target, color='blue', label="Given Locations")

    ax.set_title("Optimized 3D Missile Trajectory with Drag")
    ax.set_xlabel("Horizontal Distance X (m)")
    ax.set_ylabel("Horizontal Distance Y (m)")
    # ax.set_zlabel("Vertical Distance Z (m)")
    ax.grid(True)
    ax.legend()

    plt.show()


    # given_coords = get_cartezian_coordinates('With ID/Target bank data/Ashdod_with_ID.csv', 20)
    # print(type(given_coords))
    # given_coords = data[0]
    # answers = {}
    # for rocket in rockets:
    #     friction = rockets[rocket]['friction']
    #     mass = rockets[rocket]['mass']
    #     initialVel = rockets[rocket]['initialVel']
    #     A = rockets[rocket]['A']
    #     # Initial guess for the optimization: [x0, y0, z0, v0, angle_xy, angle_z,t,m]
    #     initial_guess = [given_coords[0][0], given_coords[0][1], given_coords[0][2], initialVel, 45,
    #                      30]  # initial guess
    #     # Fit the trajectory to the given coordinates
    #     t_eval, optimized_x, optimized_y, optimized_z, optimized_initial_conditions = fit_trajectory_to_coordinates(
    #         given_coords, initial_guess)
    #     er = error_function(optimized_initial_conditions, given_coords)
    #
    #     answers[rocket] = [er, t_eval, optimized_x, optimized_y, optimized_z, optimized_initial_conditions]
    #
    # minRocket = 'Qassam 4'
    # minError = answers[minRocket][0]
    # for rocket in answers:
    #     if answers[rocket][0] < minError:
    #         minError = answers[rocket][0]
    #         minRocket = rocket
    # print(f"minRocket: {minRocket}")
    #
    # er, t_eval, optimized_x, optimized_y, optimized_z, optimized_initial_conditions = answers[minRocket]
    #
    # lunchIndex = 0
    # hitIndex = 0
    # i = 0
    # while (i < len(optimized_z) - 1):
    #     if optimized_z[i] < 0 and 0 < optimized_z[i + 1]:
    #         lunchIndex = i
    #     if optimized_z[i] > 0 and 0 > optimized_z[i + 1]:
    #         hitIndex = i
    #     i = i + 1
    # lunchCoor = [optimized_x[lunchIndex] + optimized_x[lunchIndex + 1] / 2,
    #              optimized_y[lunchIndex] + optimized_y[lunchIndex + 1] / 2]
    # hitCoor = [optimized_x[hitIndex] + optimized_x[hitIndex + 1] / 2,
    #            optimized_y[hitIndex] + optimized_y[hitIndex + 1] / 2]
    #
    # print(f"lunchCoor:{lunchCoor}")
    # print(f"hitCoor:{hitCoor}")



    # this is the graph part
    # # # Fit the trajectory to the given coordinates
    # # t_eval, optimized_x, optimized_y, optimized_z, optimized_initial_conditions = fit_trajectory_to_coordinates(
    # #     given_coords, initial_guess)
    # # # print( t_eval, optimized_x, optimized_y, optimized_z)
    # # Plotting the result
    # fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # for rocket in rockets:
    #     er, t_eval, optimized_x, optimized_y, optimized_z, optimized_initial_conditions = answers[rocket]
    #     ax.plot(optimized_x, optimized_y, optimized_z, label=f"Predicted Trajectory:{rocket}",
    #             color=rockets[rocket]['color'])
    #
    # # Plot the optimized trajectory
    # # ax.plot(optimized_x, optimized_y, optimized_z, label="Predicted Trajectory", color='blue')
    #
    # # Plot the given points
    # ax.scatter(given_coords[:, 0], given_coords[:, 1], given_coords[:, 2], color='red', label="Given Locations")
    #
    # ax.set_title("Optimized 3D Missile Trajectory with Drag")
    # ax.set_xlabel("Horizontal Distance X (m)")
    # ax.set_ylabel("Horizontal Distance Y (m)")
    # ax.set_zlabel("Vertical Distance Z (m)")
    # ax.grid(True)
    # ax.legend()
    #
    # plt.show()
