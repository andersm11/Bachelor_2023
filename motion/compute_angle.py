import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sklearn.decomposition import PCA
from skimage import io
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from scipy.optimize import least_squares
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    

def find_center_point(points,head_axis = 0):
    center = np.mean(points, axis=0)
    if head_axis == 1:
        x = center[0]
        center = np.array([x,0,0])
    return center




def create_line(points,mean_point):
    points = np.array(points)

    # Find the minimum and maximum x values
    min_x = min(points)
    max_x = max(points)
    
    # Create the line points
    line_points = np.array([[min_x, mean_point[1], mean_point[2]],
                            [max_x, mean_point[1], mean_point[2]]])
    
    return line_points

def plot_points_and_vectors(points):

    
    mean_point = find_center_point(points,USEHEAD)

    x = [0] * len(points)
    x1 = x[0]
    for p in range(1,len(x)):
        x[p] = x1 + 0.3
        x1 = x[p]
        
    mean_line = create_line(x,mean_point)

    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract the y and z coordinates from the points
    y = [point[1] for point in points]
    z = [point[2] for point in points]




    time_diff = 0.3
    time_values = np.arange(0, len(points) * time_diff, time_diff)

    # Normalize time values to [0, 1]
    normalized_values = (time_values - np.min(time_values)) / (np.max(time_values) - np.min(time_values))

    # Plot the points
    ax.scatter(x, y, z, c='r', marker='o', s = 20)
    ax.scatter(x, y, min(z)*2, c='g', marker='o',s=8)
    ax.scatter(x, max(y)*2, z, c='r', marker='o',s=8)
    ax.scatter(-1, y, z, c='r', marker='o',s=8)

    # Convert RGBA color values to RGB and plot mean lines
    colors = cm.get_cmap('viridis')(normalized_values)
    colors = colors[:, :3]
    ax.plot(mean_line[:, 0], mean_line[:, 1], mean_line[:, 2], 'r')
    ax.plot(mean_line[:, 0], mean_line[:, 1], [min(z)*2,min(z)*2], 'g')
    ax.plot(mean_line[:, 0], [max(y)*2,max(y)*2], mean_line[:, 2], 'r')
    ax.plot([min(x)-1,min(x)-1], mean_line[:, 1], mean_line[:, 2], 'orange')
    

    
    # Plot the vectors with a gradient color
    for i in range(1, len(points)):
        ax.plot([x[i-1], x[i]], [y[i-1], y[i]], [z[i-1], z[i]], color=colors[i-1], alpha=0.8, linewidth=2)
        ax.plot([x[i-1], x[i]], [y[i-1], y[i]], [min(z)*2,min(z)*2], color='red', alpha=0.8, linewidth=1.2)
        ax.plot([x[i-1], x[i]], [max(y)*2,max(y)*2], [z[i-1], z[i]], color='green', alpha=0.8, linewidth=1.2)
        ax.plot([min(x)-1,min(x)-1], [y[i-1], y[i]], [z[i-1], z[i]], color='blue', alpha=0.8, linewidth=1.2)
    
    # Set labels and title
    ax.set_xlabel('Time(0.3s. increment)')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Points and Vectors')

    # Show the plot
    plt.show()
    

input = input("Use head axis as center?(y/n): " )
USEHEAD = 0
if input =="y" or input == "yes":
    USEHEAD = 1
input_path = "tail_points\\"

points = []
for t in range(1,11):
    points.append(np.load(input_path + "point" + str(t)+".npy"))


# Some cells are not alligned the same direction, so we rotate them so they align the same direction
for p in range(len(points)):
    if points[p][0] < 0:
        points[p][0] = 40
        points[p][1] = -points[p][1]


plot_points_and_vectors(points)


def calculate_rotation_angle(points, central_point):
    # Get the coordinates of the central point
    Cx, Cy, Cz = central_point
    
    # Calculate the rotational angle for each point
    rotational_angles = []
    for point in points:
        Px, Py, Pz = point
        

        relative_x = Px - Cx
        relative_y = Py - Cy
        relative_z = Pz - Cz
        
        # Calculate the rotational angle using arctan2
        angle = np.rad2deg(math.atan2(relative_z, relative_y))
        
        rotational_angles.append(angle)
    
    return rotational_angles



def calculate_angle_differences(rotational_angles):
    # Compute the angle differences (in degrees)
    angle_differences = []
    for i in range(len(rotational_angles) - 1):
        r1 = rotational_angles[i]
        r2 = rotational_angles[i+1]
        diff = r1 - r2
        if diff > 180:
            diff =  diff - 360 
        elif diff < -180:
            diff = diff + 360
        angle_differences.append(diff)
    
    return angle_differences


central_point = find_center_point(points,USEHEAD)


rotational_angles = calculate_rotation_angle(points, central_point)
angle_differences = calculate_angle_differences(rotational_angles)

print("Rotational Angles:", rotational_angles)
print("Angle Differences:", angle_differences)

#This display the correct angles as text, but visually it shows the smallest angles between points. Not used in report
def display_angles_3d(points, angles):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points
    xs = [0 for p in points]
    central_point = find_center_point(points,USEHEAD)
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    
    central_point = find_center_point(points,USEHEAD)
    central_point[0] = 0
    
    for i in range(len(xs) - 1):
        xs[i + 1] = xs[i] + 0.3
        
    ax.scatter(xs, ys, zs, c='b', marker='o')
    central_x, central_y, central_z = central_point
    ax.scatter(central_x, central_y, central_z, c='r', marker='o',s = 1)
    

    mean_line = create_line(xs,central_point)
    
    offset = 0.15
    
    # Compute the time difference between points
    time_diff = 0.3
    time_values = np.arange(0, len(xs) * time_diff, time_diff)
    
    # Normalize time values to [0, 1]
    normalized_values = (time_values - np.min(time_values)) / (np.max(time_values) - np.min(time_values))
    
    
    # Convert RGBA color values to RGB
    colors = cm.get_cmap('viridis')(normalized_values)
    colors = colors[:, :3]

    # Plot the angles as triangle-like figures
    for i in range(len(angles)):
        p1 = central_point
        p2 = (xs[i],ys[i],zs[i])
        p3 = (xs[i+1],ys[i+1],zs[i+1])
        angle = angles[i]

        # Calculate the direction vectors
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p1)

        
        triangle_scale = 1

        # Scale the direction vectors
        v1 *= triangle_scale
        v2 *= triangle_scale

        # Calculate the positions of the triangle vertices
        p1_vertex = np.array(p1)
        p2_vertex = np.array(p1) + v1
        p3_vertex = np.array(p1) + v2

        triangle_vertices = [p1_vertex, p2_vertex, p3_vertex]
        triangle_vertices[0][0] = triangle_vertices[0][0] + offset
        triangle_vertices[0][0] = triangle_vertices[0][0] #+ offset
        triangle_vertices[1][0] = triangle_vertices[1][0] #+ offset
        triangle_vertices[2][0] = triangle_vertices[2][0] #+ offset
     
        # Create a Poly3DCollection object for the triangle
        triangle = Poly3DCollection([triangle_vertices], alpha=0.5)
        triangle.set_facecolor(colors[i-1])
        ax.add_collection3d(triangle)
        
        # Calculate the position of the text label
        text_position = (p2_vertex + p3_vertex) / 2
        offset = offset + 0.3
        
        # Display the angle as text
        ax.text(text_position[0], text_position[1], text_position[2],
                '{:.1f}°'.format(angle), ha='center', va='center')

    ax.plot(mean_line[:, 0], mean_line[:, 1], mean_line[:, 2], 'r')
    ax.set_xlabel('Time(0.3s. increment)')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()

display_angles_3d(points, angle_differences)

def calculate_rotation_values(rotations):
    # Case 1: Sum of all angles ignoring signs
    sum_ignoring_signs = math.radians(sum(abs(angle) for angle in rotations))
    per_1_second_ignoring_signs = sum_ignoring_signs / 2.7
    per_0_3_seconds_ignoring_signs = per_1_second_ignoring_signs * 0.3
    per_0_025_seconds_ignoring_signs = per_1_second_ignoring_signs * 0.025

    # Case 3: The whole rotation is clockwise
    clockwise_angles = [math.radians(360 + angle) if angle < 0 else math.radians(abs(angle)) for angle in rotations]
    sum_clockwise = sum(clockwise_angles)
    per_1_second_clockwise = sum_clockwise / 2.7
    per_0_3_seconds_clockwise = per_1_second_clockwise * 0.3
    per_0_025_seconds_clockwise = per_1_second_clockwise * 0.025

    # Case 4: The whole rotation is counterclockwise
    counterclockwise_angles = [math.radians(360 - angle) if angle > 0 else math.radians(abs(angle)) for angle in rotations]
    sum_counterclockwise = sum(counterclockwise_angles)
    per_1_second_counterclockwise = sum_counterclockwise / 2.7
    per_0_3_seconds_counterclockwise = per_1_second_counterclockwise * 0.3
    per_0_025_seconds_counterclockwise = per_1_second_counterclockwise * 0.025

    return {
        "sum_ignoring_signs": sum_ignoring_signs,
        "per_1_second_ignoring_signs": per_1_second_ignoring_signs,
        "per_0_3_seconds_ignoring_signs": per_0_3_seconds_ignoring_signs,
        "per_0_025_seconds_ignoring_signs": per_0_025_seconds_ignoring_signs,
        "sum_clockwise": sum_clockwise,
        "per_1_second_clockwise": per_1_second_clockwise,
        "per_0_3_seconds_clockwise": per_0_3_seconds_clockwise,
        "per_0_025_seconds_clockwise": per_0_025_seconds_clockwise,
        "sum_counterclockwise": sum_counterclockwise,
        "per_1_second_counterclockwise": per_1_second_counterclockwise,
        "per_0_3_seconds_counterclockwise": per_0_3_seconds_counterclockwise,
        "per_0_025_seconds_counterclockwise": per_0_025_seconds_counterclockwise,
    }
    
res = calculate_rotation_values(angle_differences)
print(res)
    