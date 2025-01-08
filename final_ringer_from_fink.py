import math
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

makams_impact = {'impactdata/Ashdod_with_ID.csv': (31.77757586390034, 34.65751251836753),
          'impactdata/Kiryat_Gat_with_ID.csv': (31.602089287486198, 34.74535762921831),
          'impactdata/Ofakim_with_ID.csv': (31.302709659709315, 34.59685294800365),
          'impactdata/Tseelim_with_ID.csv': (31.20184656499955, 34.52669152933695),
          'impactdata/Meron_with_ID.csv': (33.00023023451869, 35.404698698883585),
          'impactdata/YABA_with_ID.csv': (30.65361041190953, 34.783379139342955),
          'impactdata/Modiin_with_ID.csv': (31.891980958022323, 34.99481765229601),
          'impactdata/Gosh_Dan_with_ID.csv': (32.105913486777084, 34.78624983651992),
          'impactdata/Carmel_with_ID.csv': (32.65365306190331, 35.03028065430696)}

makams_target = {'targetdata/Ashdod_with_IDtarget.csv': (31.77757586390034, 34.65751251836753),
          'targetdata/Kiryat_Gat_with_IDtarget.csv': (31.602089287486198, 34.74535762921831),
          'targetdata/Ofakim_with_IDtarget.csv': (31.302709659709315, 34.59685294800365),
          'targetdata/Tseelim_with_IDtarget.csv': (31.20184656499955, 34.52669152933695),
          'targetdata/Meron_with_IDtarget.csv': (33.00023023451869, 35.404698698883585),
          'targetdata/YABA_with_IDtarget.csv': (30.65361041190953, 34.783379139342955),
          'targetdata/Modiin_with_IDtarget.csv': (31.891980958022323, 34.99481765229601),
          'targetdata/Gosh_Dan_with_IDtarget.csv': (32.105913486777084, 34.78624983651992),
          'targetdata/Carmel_with_IDtarget.csv': (32.65365306190331, 35.03028065430696)}

TO_RAD = np.pi / 180
EARTH_RADIUS = 6371000


def split_data_by_id(file_paths, makams):
    """
    Reads multiple CSV files and splits the combined data by unique ID values.

    Parameters:
        file_paths (list of str): List of file paths to CSV files.

    Returns:
        list of pandas.DataFrame: A list of DataFrames, each containing all data
                                  from the input files split by unique ID.
    """
    # Read and combine all files into a single DataFrame
    combined_data = pd.DataFrame()
    for file_path in file_paths:
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            add_column_of_global_coord(data, makams[file_path][0], makams[file_path][1])
            combined_data = pd.concat([combined_data, data], ignore_index=True)
        else:
            print(f"File not found: {file_path}")

    # Ensure the ID column exists
    if 'ID' not in combined_data.columns:
        raise ValueError("The 'ID' column is missing in the input files.")

    # Split data by unique IDs
    data_by_id = [group for _, group in combined_data.groupby('ID')]

    return data_by_id


def add_column_of_global_coord(df, lat, lon):
    column_x = []
    column_y = []
    column_z = []
    for index, row in df.iterrows():
        x, y, z = radar_measurement_to_xyz(lat, lon, row['range'], row['elevation'], row['azimuth'])
        z, x, y = convert_to_ashdod(x, y, z)
        column_x.append(x)
        column_y.append(y)
        column_z.append(z)

    df['x'] = column_x
    df['y'] = column_y
    df['z'] = column_z


def spherical_to_global(phi, theta, r, theta0, el):
    """
    ממירה קורדינאטות כדוריות יחסיות לקורדינאטות גלובליות קרטזיות.

    פרמטרים:
    lat -- קו רוחב של הנקודה הגאוגרפית (במעלות)
    lon -- קו אורך של הנקודה הגאוגרפית (במעלות)
    r -- מרחק רדיאלי (באותן יחידות כמו רדיוס כדור הארץ)
    theta -- הזווית מקו הזניט (במעלות)
    phi -- הזווית היחסית במישור האופקי (במעלות)

    מחזירה:
    tuple של (x, y, z) במערכת הקואורדינטות הגלובליות
    """
    R = 6371000

    x0 = (R * math.cos(phi) - r * math.cos(el) * math.cos(theta0) * math.sin(phi)) * math.cos(theta) - r * math.cos(
        el) * math.sin(theta0) * math.sin(theta)
    y0 = (R * math.cos(phi) - r * math.cos(el) * math.cos(theta0) * math.sin(phi)) * math.sin(theta) + r * math.cos(
        el) * math.sin(theta0) * math.cos(theta)
    z0 = R * math.sin(phi) + r * math.cos(el) * math.cos(theta0) * math.cos(phi)
    return x0, y0, z0

    # # המרת מעלות לרדיאנים
    # lat_rad = math.radians(lat)
    # lon_rad = math.radians(lon)
    # theta_rad = math.radians(theta)
    # el_rad = math.radians(el)
    #
    # # מיקום הנקודה הגאוגרפית במערכת הקרטזית
    # x0 = R_earth * math.cos(lat_rad) * math.cos(lon_rad)
    # y0 = R_earth * math.cos(lat_rad) * math.sin(lon_rad)
    # z0 = R_earth * math.sin(lat_rad)
    #
    # # המיקום היחסי במערכת הקרטזית
    # dx = r * math.sin(theta_rad) * math.cos(phi_rad)
    # dy = r * math.sin(theta_rad) * math.sin(phi_rad)
    # dz = r * math.cos(theta_rad)
    #
    # # המרת המיקום היחסי לגלובלי
    # x = x0 + dx
    # y = y0 + dy
    # z = z0 + dz
    #
    # return x, y, z


def radar_measurement_to_xyz(latitude: float, longitude: float, radius: float, elevation: float, azimuth: float):
    """
    transform a radar measurement into a point in xyz. the origin of xyz is at the center of the earth, z is north
    x points to lat, lon = 0, 0
    :param latitude, longitude: radar's lat and lon [DEG]
    :param radius, elevation, azimuth: measured data point
    :return: point in xyz
    """

    latitude *= TO_RAD
    longitude *= TO_RAD
    azimuth *= TO_RAD
    elevation *= TO_RAD

    radar_xyz = EARTH_RADIUS * np.array([np.cos(latitude) * np.cos(longitude),
                                         np.cos(latitude) * np.sin(longitude),
                                         np.sin(latitude)])

    east = radius * np.cos(elevation) * np.sin(azimuth)
    north = radius * np.cos(elevation) * np.cos(azimuth)
    altitude = radius * np.sin(elevation)

    east_hat = np.array([-np.sin(longitude),
                         np.cos(longitude),
                         0])
    north_hat = np.array([np.sin(latitude) * -1 * np.cos(longitude),
                          np.sin(latitude) * -1 * np.sin(longitude),
                          np.cos(latitude)])
    altitude_hat = np.array([np.cos(latitude) * np.cos(longitude),
                             np.cos(latitude) * np.sin(longitude),
                             np.sin(latitude)])

    return radar_xyz + east * east_hat + north * north_hat + altitude * altitude_hat


def convert_to_ashdod(x, y, z):
    theta = (90 - 31.77757586390034) * TO_RAD
    phi = 34.65751251836753 * TO_RAD
    R = EARTH_RADIUS
    x0 = R * math.sin(theta) * math.cos(phi)
    y0 = R * math.sin(theta) * math.sin(phi)
    z0 = R * math.cos(theta)

    v = np.array([x - x0, y - y0, z - z0])
    vr = np.array([[math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)]
                      , [math.cos(theta) * math.cos(phi), math.cos(theta) * math.sin(phi), -math.sin(theta)]
                      , [-math.sin(phi), math.cos(phi), 0]]) @ v

    return vr

def convert_from_ashdod(x, y, z):

    theta = (90 - 31.77757586390034) * pi / 180
    phi = 34.65751251836753 * pi / 180
    R = 6371000
    x0 = R * sin(theta) * cos(phi)
    y0 = R * sin(theta) * sin(phi)
    z0 = R*cos(theta)

    vr = np.array([[sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]
    , [cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta)]
    , [-sin(phi), cos(phi), 0]]).T @ np.array([z, x, y])

    v =  vr + np.array([x0, y0, z0 ])

    return v

def xyz_to_lat_and_lon(x, y, z):
    lat = math.asin(z / math.sqrt(z ** 2 + x ** 2 + y ** 2)) / TO_RAD
    lon = math.asin(y / math.sqrt(y ** 2 + x ** 2)) / TO_RAD
    return lat, lon


def plot_3d_locations(x, y, z):
    """
    Plots a list of three-dimensional locations on a 3D graph.

    Parameters:
        x,y,z: A list of (x, y, z) coordinates.

    Returns:
        None
    """
    # Extract x, y, z coordinates
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(x, y, z, c='b', marker='o', label='Locations')

    # Set labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('3D Locations Plot')

    # Show legend and grid
    ax.legend()
    ax.grid(True)

    # Display the plot
    plt.show()


def convert_to_cartesian_and_time(df):
    return pd.DataFrame({'time': df['time'], 'x': df['x'], 'y': df['y'], 'z': df['z'], "ID": df['ID']})



# # # Example usage
# file_paths = makams.keys()
# # Replace with actual file paths
#
# print(file_paths)
#
# split_data = [convert_to_cartesian_and_time(rocket) for rocket in split_data_by_id(file_paths)]

def get_list_of_df(impact:bool):
    if impact:
        makams = makams_impact
    else:
        makams = makams_target
    file_paths = makams.keys()
    # Replace with actual file paths

    print(file_paths)

    return [convert_to_cartesian_and_time(rocket) for rocket in split_data_by_id(file_paths, makams)]

# print(split_data)
#
#
# print(split_data[0])
#
#
# for rocket in split_data:
#     plot_3d_locations(rocket['x'], rocket['y'], rocket['z'])

# x,y,z = radar_measurement_to_xyz(makams[list(makams.keys())[0]][0], makams[list(makams.keys())[0]][1], 0,0,0)
# print(xyz_to_lat_and_lon(x,y,z))
if __name__ == '__main__':
    splt1 = get_list_of_df(False)
    for rocket in splt1:
        plot_3d_locations(rocket['x'], rocket['y'], rocket['z'])


