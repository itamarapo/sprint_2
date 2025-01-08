import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# Constants (Example values)
g = 9.81  # gravity acceleration in m/s^2
C_d = 2.28  # drag coefficient (dimensionless)
A = 0.01  # cross-sectional area of the rocket in m^2 (example)
rho = 1.204  # air density at sea level in kg/m^3
m0 = 50  # initial total mass (rocket + fuel) in kg
m_fuel = 30  # fuel mass in kg
v_e = 1200  # exhaust velocity of the rocket in m/s
burn_time = 0.5  # fuel burn time in seconds

# Derived parameters
fuel_burn_rate = m_fuel / burn_time  # fuel burn rate (kg/s)


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
    coordinates = np.stack((x, y, z))
    return coordinates, time_data


# Known positions at specific times (example values)
known_positions, times = get_cartezian_coordinates("impactdata/Ashdod_with_ID.csv", 5)
# times = np.array([0, 20, 40, 60, 80, 100])  # Time in seconds
# known_positions = np.array([
#     [0, 0, 0],  # Position at t=0
#     [100, 150, 200],  # Position at t=20
#     [300, 450, 600],  # Position at t=40
#     [500, 750, 1000],  # Position at t=60
#     [700, 1000, 1300],  # Position at t=80
#     [900, 1200, 1600],  # Position at t=100
# ])


# Define the system of equations (motion + fuel consumption)
def rocket_eq(t, y, m0, m_fuel, fuel_burn_rate):
    pos = y[:3]  # position (x, y, z)
    vel = y[3:6]  # velocity (vx, vy, vz)

    # Calculate the rocket's mass at time t, ensuring it doesn't go below the mass without fuel
    mass = m0 - min(fuel_burn_rate * t, m_fuel)

    # Compute the drag force: F_drag = 0.5 * C_d * A * rho * v^2 * unit_vector
    speed = np.linalg.norm(vel)
    drag_force = -0.5 * C_d * A * rho * speed ** 2 * (vel / speed)

    # Compute the gravitational force: F_gravity = -m * g
    gravity_force = np.array([0, 0, -mass * g])

    # Net acceleration = (gravity + drag) / mass
    net_force = gravity_force + drag_force
    acceleration = net_force / mass

    # Return the derivatives [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt]
    dydt = np.concatenate([vel, acceleration])
    return dydt


# Define the objective function for optimization
def objective(params, times, known_positions, m0, m_fuel, fuel_burn_rate):
    initial_position = params[:3]
    initial_velocity = params[3:6]
    t0 = params[6]  # Launch time is the 7th parameter

    # Adjust times by subtracting the launch time (to shift the timeline)
    adjusted_times = times - t0

    # Set the initial conditions for the system
    initial_conditions = np.concatenate([initial_position, initial_velocity])

    # Solve the system of equations using solve_ivp (RK45)
    solution = solve_ivp(rocket_eq, (adjusted_times[0], adjusted_times[-1]), initial_conditions, t_eval=adjusted_times,
                         args=(m0, m_fuel, fuel_burn_rate), method='RK45')

    # Calculate the difference between the known positions and the model's predicted positions
    predicted_positions = solution.y[:3, :]  # [x, y, z] positions from the model

    # Compute the residuals (differences) between the known and predicted positions
    residuals = predicted_positions - known_positions.T

    # Return the residuals flattened (for least squares)
    return residuals.flatten()


# Initial guess for the optimization (initial position, velocity, and launch time)
initial_guess = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # initial guess: [x0, y0, z0, vx0, vy0, vz0, t0]

# Optimize to find the best-fit initial conditions and launch time
result = least_squares(objective, initial_guess, args=(times, known_positions, m0, m_fuel, fuel_burn_rate))

# Extract the optimized initial conditions and launch time
optimized_position = result.x[:3]
optimized_velocity = result.x[3:6]
optimized_launch_time = result.x[6]

print(f"Optimized Initial Position: {optimized_position}")
print(f"Optimized Initial Velocity: {optimized_velocity}")
print(f"Optimized Launch Time: {optimized_launch_time}")

# Solve the system with the optimized initial conditions and launch time
optimized_initial_conditions = np.concatenate([optimized_position, optimized_velocity])
adjusted_times = times - optimized_launch_time  # Adjust the times based on the optimized launch time
solution_optimized = solve_ivp(rocket_eq, (adjusted_times[0], adjusted_times[-1]), optimized_initial_conditions,
                               t_eval=adjusted_times, args=(m0, m_fuel, fuel_burn_rate), method='RK45')

# Plot the results
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(solution_optimized.y[0], solution_optimized.y[1], solution_optimized.y[2], label="Optimized Rocket Trajectory")
ax.scatter(known_positions[:, 0], known_positions[:, 1], known_positions[:, 2], color='r', label="Known Positions")
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('Rocket Trajectory with Optimized Initial Conditions and Launch Time')
plt.legend()
plt.show()
