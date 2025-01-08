import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Constants
R = 6371000  # Earth's radius in meters (mean)

# Radar 1's geographical location
radar_lat1 = 31.77757586390034  # Latitude of Radar 1 (degrees)
radar_lon1 = 34.65751251836753  # Longitude of Radar 1 (degrees)
radar_alt1 = 0  # Altitude of Radar 1 (meters)

file_path1 = 'Target bank data/Ashdod_with_ID.csv'  # Path for Radar 1's CSV file

# Read Radar 1's CSV file
df1 = pd.read_csv(file_path1)
df1 = df1[df1['ID'] == 1]

# Extracting range, elevation, and azimuth for Radar 1
range_data1 = df1['range'].to_numpy()
elevation_data1 = df1['elevation'].to_numpy()
azimuth_data1 = df1['azimuth'].to_numpy()

# Convert degrees to radians for Radar 1
elevation_rad1 = np.radians(elevation_data1)
azimuth_rad1 = np.radians(azimuth_data1)

# Convert spherical to Cartesian coordinates for Radar 1
x_arr1 = range_data1 * np.cos(elevation_rad1) * np.cos(azimuth_rad1)
y_arr1 = range_data1 * np.cos(elevation_rad1) * np.sin(azimuth_rad1)
z_arr1 = range_data1 * np.sin(elevation_rad1)

# For Radar 2
radar_lat2 = 32.65365306190331  # Latitude of Radar 2 (degrees)
radar_lon2 = 35.03028065430696  # Longitude of Radar 2 (degrees)
radar_alt2 = 0  # Altitude of Radar 2 (meters)

file_path2 = 'Target bank data/Carmel_with_ID.csv'  # Path for Radar 2's CSV file

# Read Radar 2's CSV file
df2 = pd.read_csv(file_path2)
df2 = df2[df2['ID'] == 1]  # Corrected here, using df2 for filtering

# Extracting range, elevation, and azimuth for Radar 2
range_data2 = df2['range'].to_numpy()
elevation_data2 = df2['elevation'].to_numpy()
azimuth_data2 = df2['azimuth'].to_numpy()

# Convert degrees to radians for Radar 2
elevation_rad2 = np.radians(elevation_data2)
azimuth_rad2 = np.radians(azimuth_data2)

# Convert spherical to Cartesian coordinates for Radar 2
x_arr2 = range_data2 * np.cos(elevation_rad2) * np.cos(azimuth_rad2)
y_arr2 = range_data2 * np.cos(elevation_rad2) * np.sin(azimuth_rad2)
z_arr2 = range_data2 * np.sin(elevation_rad2)

# For Radar 3 ("Gosh_Dan")
radar_lat3 = 32.105913486777084  # Latitude of "Gosh_Dan" (degrees)
radar_lon3 = 34.78624983651992  # Longitude of "Gosh_Dan" (degrees)
radar_alt3 = 0  # Altitude of "Gosh_Dan" (meters)

file_path3 = 'Target bank data/Gosh_Dan_with_ID.csv'  # Path for Radar 3's CSV file

# Read Radar 3's CSV file
df3 = pd.read_csv(file_path3)
df3 = df3[df3['ID'] == 1]  # Corrected here, using df3 for filtering

# Extracting range, elevation, and azimuth for Radar 3
range_data3 = df3['range'].to_numpy()
elevation_data3 = df3['elevation'].to_numpy()
azimuth_data3 = df3['azimuth'].to_numpy()

# Convert degrees to radians for Radar 3
elevation_rad3 = np.radians(elevation_data3)
azimuth_rad3 = np.radians(azimuth_data3)

# Convert spherical to Cartesian coordinates for Radar 3
x_arr3 = range_data3 * np.cos(elevation_rad3) * np.cos(azimuth_rad3)
y_arr3 = range_data3 * np.cos(elevation_rad3) * np.sin(azimuth_rad3)
z_arr3 = range_data3 * np.sin(elevation_rad3)

# Function to convert from geographical to Cartesian coordinates relative to Radar 1
def geo_to_cartesian(lat, lon, alt, radar_lat1, radar_lon1, radar_alt1):
    # Convert geographical coordinates of another radar to Cartesian relative to Radar 1
    delta_lat = np.radians(lat - radar_lat1) * R
    delta_lon = np.radians(lon - radar_lon1) * R * np.cos(np.radians(radar_lat1))
    delta_alt = alt - radar_alt1
    return delta_lon, delta_lat, delta_alt

# Convert Radar 1 location to Cartesian coordinates (origin, so it's just (0, 0, 0))
x1, y1, z1 = 631778, 167644, 0

# Convert Radar 2's geographical location to Cartesian relative to Radar 1
x2, y2, z2 = 728811,203111, 0

# Convert Radar 3's geographical location to Cartesian relative to Radar 1
x3, y3, z3 = 668132,179979, 0

# Subtract Radar 1's Cartesian coordinates from all points in the racket path of Radar 1
x_arr1_rel, y_arr1_rel, z_arr1_rel = x_arr1 - x1, y_arr1 - y1, z_arr1 - z1

# Subtract Radar 2's Cartesian coordinates from all points in the racket path of Radar 2
x_arr2_rel, y_arr2_rel, z_arr2_rel = x_arr2 - x2, y_arr2 - y2, z_arr2 - z2

# Subtract Radar 3's Cartesian coordinates from all points in the racket path of Radar 3
x_arr3_rel, y_arr3_rel, z_arr3_rel = x_arr3 - x3, y_arr3 - y3, z_arr3 - z3


# 3D Cartesian Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot racket path from Radar 1 (origin)
ax.plot(x_arr1_rel, y_arr1_rel, z_arr1_rel, label="Racket Path from Radar 1", marker='o', linestyle='-', markersize=5)

# Plot racket path from Radar 2 (relative to Radar 1)
ax.plot(x_arr2_rel, y_arr2_rel, z_arr2_rel, label="Racket Path from Radar 2", marker='x', linestyle='-', markersize=5)

# Plot racket path from Radar 3 (relative to Radar 1)
ax.plot(x_arr3_rel, y_arr3_rel, z_arr3_rel, label="Racket Path from Gosh_Dan", marker='^', linestyle='-', markersize=5)

# # Mark Radar 1 location (origin)
# ax.scatter(x1, y1, z1, color='red', label="Radar 1 Location", s=50)
#
# # Mark Radar 2 location (relative to Radar 1)
# # Use x2, y2, z2 directly without indexing
# ax.scatter(x2, y2, z2, color='blue', label="Radar 2 Location", s=50)
#
#
# # Mark Radar 3 location (relative to Radar 1)
# ax.scatter(x3, y3, z3, color='green', label="Gosh_Dan Location", s=50)

# Add labels
ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")
ax.set_zlabel("Z (meters)")
ax.set_title("Racket Paths in Cartesian Coordinates (Radar 1 as Origin)")
ax.legend()

# Show the 3D plot
plt.show()
