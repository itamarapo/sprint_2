import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def find_threshold(df, i, threshold0 = 200 ,min_dots = 50):
    threshold = threshold0
    n = len(df[(df["ID"] == i) & (df["range_uncertainty"] < threshold)])
    while n > min_dots:
        threshold -= 5
        n = len(df[(df["ID"] == i) & (df["range_uncertainty"] < threshold)])
    return threshold

def geodetic_to_cartesian(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """
    Converts a location in geodetic coordinates (latitude, longitude, altitude)
    to Cartesian coordinates (x, y, z) relative to a specific reference location.

    Parameters:
        lat (float): Latitude of the location (in degrees).
        lon (float): Longitude of the location (in degrees).
        alt (float): Altitude of the location (in meters).
        ref_lat (float): Latitude of the reference location (in degrees).
        ref_lon (float): Longitude of the reference location (in degrees).
        ref_alt (float): Altitude of the reference location (in meters).

    Returns:
        tuple: Cartesian coordinates (x, y, z) relative to the reference location.
    """
    # WGS84 ellipsoid constants
    a = 6378137.0  # Semi-major axis (in meters)
    f = 1 / 298.257223563  # Flattening
    e2 = f * (2 - f)  # Square of eccentricity

    # Helper function to convert lat/lon/alt to ECEF coordinates
    def geodetic_to_ecef(lat, lon, alt):
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        # Prime vertical radius of curvature
        N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)

        # Calculate ECEF coordinates
        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - e2) + alt) * np.sin(lat_rad)

        return np.array([x, y, z])

    # Convert reference location to ECEF
    ref_ecef = geodetic_to_ecef(ref_lat, ref_lon, ref_alt)

    # Convert target location to ECEF
    target_ecef = geodetic_to_ecef(lat, lon, alt)

    # Calculate relative Cartesian coordinates
    relative_cartesian = target_ecef - ref_ecef

    return tuple(relative_cartesian)





from mpl_toolkits.mplot3d import Axes3D

# Constants
R = 6371000  # Earth's radius in meters (mean)

# Read the radars.csv file to get the radar locations
radar_data = pd.read_csv("radars.csv")

# Read and process racket data for all radars
dot = 0
threshold = 50
for i in range(1,100):
    radar_paths = []
    dot_sum = 0
    for _, radar in radar_data.iterrows():
        radar_name = radar["Radar"]
        radar_lat = radar["X"]
        radar_lon = radar["Y"]
        radar_alt = 0  # Assuming altitudes are 0 for simplicity

        # Read the corresponding CSV file for this radar
        file_path = f"Target bank data/{radar_name}_with_ID.csv"
        df = pd.read_csv(file_path)
        threshold = find_threshold(df,i)
        df = df[(df["ID"] == i) & (df["range_uncertainty"] < threshold)]
        dot_sum+=len(df)
        # Extract range, elevation, and azimuth
        range_data = df["range"].to_numpy()
        elevation_data = np.radians(df["elevation"].to_numpy())  # Convert to radians
        azimuth_data = np.radians(df["azimuth"].to_numpy())  # Convert to radians

        # Convert spherical to Cartesian coordinates for the radar
        x_arr = range_data * np.cos(elevation_data) * np.cos(azimuth_data)
        y_arr = range_data * np.cos(elevation_data) * np.sin(azimuth_data)
        z_arr = range_data * np.sin(elevation_data)

        # Save the radar's Cartesian data and metadata
        radar_paths.append({
            "name": radar_name,
            "x_arr": x_arr,
            "y_arr": y_arr,
            "z_arr": z_arr,
            "x_radar": radar["X"],
            "y_radar": radar["Y"],
            "z_radar": radar_alt
        })
    print(f"""rocket id: {i}
          max error: {threshold}\ndot amount: {dot_sum}""")


    # Adjust all radar paths relative to the reference radar
    for path in radar_paths:
        path["x_arr_rel"] = path["x_arr"] + (path["x_radar"])
        path["y_arr_rel"] = path["y_arr"] + (path["y_radar"])
        path["z_arr_rel"] = path["z_arr"] + (path["z_radar"])

    # 3D Cartesian Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot racket paths for all radars
    for path in radar_paths:
        ax.plot(
            path["x_arr_rel"],
            path["y_arr_rel"],
            path["z_arr_rel"],
            label=f"Racket Path from {path['name']}",
            linestyle="-",
            marker="o",
            markersize=5
        )

    # Add labels and title
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Z (meters)")
    ax.set_title("Racket Paths in Cartesian Coordinates, Racket " + str(i) + " Threshold " + str(threshold))
    ax.legend()

    # Show the 3D plot
    plt.show()
