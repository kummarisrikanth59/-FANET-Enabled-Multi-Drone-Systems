import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as patches
import time
from queue import Queue
from matplotlib import gridspec
import matplotlib.colors as mcolors
import os

# Constants - All distances in kilometers
NUM_DRONES = 10
NUM_CAMERAS = NUM_DRONES  # One camera per drone
AREA_SIZE = 20  # km
COMM_RANGE = 6  # km
OBSTACLE_RADIUS = 0.5  # km
DYNAMIC_OBSTACLE_RADIUS = 0.7  # km
CAMERA_RADIUS = 2.0  # km
WILDLIFE_RADIUS = 0.3  # km
BIKER_RADIUS = 0.2  # km
CIRCLE_RADIUS = 4.5  # km
FIGURE8_WIDTH = 4.0 # km
FIGURE8_HEIGHT = 3.0 # km
FORMATION_SPEED = 0.02  # Speed of rotation
ALTITUDE_RANGE = (2.0, 5.0)  # km
SIMULATION_STEPS = 1000  # 500 steps for Circle, 500 for Figure-8

# Static obstacles
OBSTACLES = np.array([
    [-6.0, 4.0], [5.0, 5.0], [-5.0, -5.0], [6.0, -4.0], [3.0, 0.0], [0.5, -1.0], [0.0, 6.0], [-7.0, -2.0], [2.0, 7.0],
    [-4.0, 7.0], [7.0, 1.0], [-1.0, -7.0], [-8.0, 3.0], [4.0, -6.0], [1.0, 8.0], [-2.0, -3.0], [6.0, 2.0], [-5.0, 1.0],
    [3.0, -5.0], [-7.0, -1.0], [5.0, -3.0], [-3.0, 5.0],
    [9.0, 9.0], [8.0, 8.0], [7.0, 7.0], [9.0, -9.0], [8.0, -8.0], [7.0, -7.0],
    [7.5, 7.5], [-7.5, -7.5], [-7.5, 7.5], [7.5, -7.5], [-9.0, 9.0], [-8.0, 8.0],
    [-8.0, -8.0], [-7.0, 7.0], [5.0, 7.5], [-5.0, 7.5], [-5.0, -7.5], [5.0, -7.5]
])
OBSTACLE_RADII = np.array([0.5, 0.6, 0.4, 0.7, 0.5, 0.6, 0.8, 0.4, 0.3, 0.5, 0.6, 0.4, 0.5, 0.7, 0.6, 0.4, 0.5, 0.6, 0.8, 0.4, 0.3, 0.5] +
                          [0.5] * 18)

# Dynamic obstacles (birds)
NUM_BIRDS = 7
np.random.seed(42)
birds = np.random.uniform(-AREA_SIZE/2 + DYNAMIC_OBSTACLE_RADIUS, 
                         AREA_SIZE/2 - DYNAMIC_OBSTACLE_RADIUS, 
                         size=(NUM_BIRDS, 2))
bird_velocities = np.random.uniform(-0.03, 0.03, size=(NUM_BIRDS, 2))

# Wildlife
NUM_WILDLIFE = 24
wildlife = np.random.uniform(-AREA_SIZE/2 + WILDLIFE_RADIUS, 
                            AREA_SIZE/2 - WILDLIFE_RADIUS, 
                            size=(NUM_WILDLIFE, 2))
wildlife_velocities = np.random.uniform(-0.02, 0.02, size=(NUM_WILDLIFE, 2))
wildlife_names = ['Tiger'] * 8 + ['Lion'] * 8 + ['Deer'] * 8

# Bikers
NUM_BIKERS = 8
bikers = np.random.uniform(-AREA_SIZE/2 + BIKER_RADIUS, 
                          AREA_SIZE/2 - BIKER_RADIUS, 
                          size=(NUM_BIKERS, 2))
biker_velocities = np.random.uniform(-0.05, 0.05, size=(NUM_BIKERS, 2))

# Icon directory
ICON_DIR = r"D:\VS codes\matlab\icons"

def load_icon(img_path):
    if not os.path.exists(img_path):
        base_path = os.path.splitext(img_path)[0]
        for ext in ['.png', '.jpg', '.jpeg']:
            test_path = base_path + ext
            if os.path.exists(test_path):
                img_path = test_path
                break
        else:
            print(f"Warning: Icon not found at {img_path} or with alternate extensions")
            return None
    print(f"Loading icon from: {img_path}")
    try:
        icon = plt.imread(img_path)
        if not isinstance(icon, np.ndarray) or icon.dtype not in [np.uint8, np.float32, np.float64]:
            print(f"Warning: Invalid image data type for {img_path}. Skipping.")
            return None
        return icon
    except Exception as e:
        print(f"Error loading icon from {img_path}: {e}")
        return None

def load_drone_icon():
    img_path = os.path.join(ICON_DIR, "drone.png")
    icon = load_icon(img_path)
    return OffsetImage(icon, zoom=0.125) if icon is not None else None

def load_obstacle_icon():
    img_path = os.path.join(ICON_DIR, "tree.png")
    icon = load_icon(img_path)
    return icon if icon is not None else None

def load_bird_icon():
    img_path = os.path.join(ICON_DIR, "bird.png")
    icon = load_icon(img_path)
    return icon if icon is not None else None

def load_animal_icon(wildlife_idx):
    if wildlife_idx < 8:
        img_name = "tiger"
    elif wildlife_idx < 16:
        img_name = "lion"
    else:
        img_name = "deer"
    img_path = os.path.join(ICON_DIR, "animals", f"{img_name}.png")
    icon = load_icon(img_path)
    return OffsetImage(icon, zoom=0.04) if icon is not None else None

def load_biker_icon():
    img_path = os.path.join(ICON_DIR, "animals", "bikers.jpg")
    icon = load_icon(img_path)
    return OffsetImage(icon, zoom=0.05) if icon is not None else None

# Drone controller for both formations
class DroneFormationController:
    def __init__(self, num_drones):
        self.num_drones = num_drones
    
    def get_circle_pattern(self, t, center=(0, 0, 0), clockwise=True):
        positions = np.zeros((self.num_drones, 3))
        angular_spacing = 2 * np.pi / self.num_drones
        rotation_offset = t * FORMATION_SPEED * (1 if clockwise else -1)
        for i in range(self.num_drones):
            angle = i * angular_spacing + rotation_offset
            x = center[0] + CIRCLE_RADIUS * np.cos(angle)
            y = center[1] + CIRCLE_RADIUS * np.sin(angle)
            z = center[2] + np.interp(i, [0, self.num_drones-1], ALTITUDE_RANGE)
            positions[i] = [x, y, z]
        return positions
        
    def get_figure8_pattern(self, t):
        positions = np.zeros((self.num_drones, 3))
        base_time = t * FORMATION_SPEED
        for i in range(self.num_drones):
            phase = base_time + (i * 2 * np.pi / self.num_drones)
            cos_phase = np.cos(phase)
            sin_phase = np.sin(phase)
            denominator = 1 + sin_phase**2
            x = FIGURE8_WIDTH * cos_phase / denominator
            y = FIGURE8_HEIGHT * sin_phase * cos_phase / denominator
            z = np.interp(i, [0, self.num_drones-1], ALTITUDE_RANGE)
            positions[i] = [x, y, z]
        return positions

formation_controller = DroneFormationController(NUM_DRONES)

# Initialize drone swarm
np.random.seed(42)
positions = np.hstack((np.random.uniform(-3, 3, size=(NUM_DRONES, 2)), 
                       np.random.uniform(ALTITUDE_RANGE[0], ALTITUDE_RANGE[1], size=(NUM_DRONES, 1))))
velocities = np.zeros((NUM_DRONES, 3))
target_positions = np.zeros((NUM_DRONES, 3))

# Simulation state
formation_time = 0

# Set up plot
fig = plt.figure(figsize=(36, 12))
gs = gridspec.GridSpec(1, 3, width_ratios=[3, 1, 1])
ax_main = plt.subplot(gs[0])
ax_info = plt.subplot(gs[1])
ax_altitude = plt.subplot(gs[2])
plt.ion()

def advanced_movement_control(current_pos, current_vel, target_pos, max_speed=0.3, arrival_threshold=0.1):
    error = target_pos - current_pos
    distance = np.linalg.norm(error)
    if distance < arrival_threshold:
        desired_velocity = error * 0.5
    else:
        desired_velocity = error * min(1.0, max_speed / max(distance, 0.1))
    velocity_change = (desired_velocity - current_vel) * 0.2
    new_velocity = current_vel + velocity_change
    speed = np.linalg.norm(new_velocity)
    if speed > max_speed:
        new_velocity = new_velocity / speed * max_speed
    new_velocity *= 0.9
    return current_pos + new_velocity, new_velocity

def enhanced_obstacle_avoidance(pos, target, static_obstacles, static_radii, birds, bird_radii, wildlife, wildlife_radii, bikers, biker_radii, avoidance_strength=2.0):
    avoidance_vector = np.zeros(3)
    for i, obstacle in enumerate(static_obstacles):
        to_obstacle = pos[:2] - obstacle
        distance = np.linalg.norm(to_obstacle)
        radius = static_radii[i]
        if distance < radius + avoidance_strength and distance > 0.01:
            repulsion_magnitude = avoidance_strength / (distance - radius + 0.1)**2
            avoidance_vector[:2] += (to_obstacle / distance) * repulsion_magnitude
    for i, obstacle in enumerate(birds):
        to_obstacle = pos[:2] - obstacle
        distance = np.linalg.norm(to_obstacle)
        if distance < bird_radii + avoidance_strength and distance > 0.01:
            repulsion_magnitude = avoidance_strength / (distance - bird_radii + 0.1)**2
            avoidance_vector[:2] += (to_obstacle / distance) * repulsion_magnitude
    for i, obstacle in enumerate(wildlife):
        to_obstacle = pos[:2] - obstacle
        distance = np.linalg.norm(to_obstacle)
        if distance < wildlife_radii + avoidance_strength and distance > 0.01:
            repulsion_magnitude = avoidance_strength / (distance - wildlife_radii + 0.1)**2
            avoidance_vector[:2] += (to_obstacle / distance) * repulsion_magnitude
    for i, obstacle in enumerate(bikers):
        to_obstacle = pos[:2] - obstacle
        distance = np.linalg.norm(to_obstacle)
        if distance < biker_radii + avoidance_strength and distance > 0.01:
            repulsion_magnitude = avoidance_strength / (distance - biker_radii + 0.1)**2
            avoidance_vector[:2] += (to_obstacle / distance) * repulsion_magnitude
    if np.linalg.norm(avoidance_vector[:2]) > 0:
        target[:2] += avoidance_vector[:2] * 0.5
    return target

def update_dynamic_obstacles():
    global birds, bird_velocities, wildlife, wildlife_velocities, bikers, biker_velocities
    for i in range(NUM_BIRDS):
        birds[i] += bird_velocities[i]
        for dim in range(2):
            if birds[i, dim] < -AREA_SIZE/2 + DYNAMIC_OBSTACLE_RADIUS:
                birds[i, dim] = -AREA_SIZE/2 + DYNAMIC_OBSTACLE_RADIUS
                bird_velocities[i, dim] *= -1
            elif birds[i, dim] > AREA_SIZE/2 - DYNAMIC_OBSTACLE_RADIUS:
                birds[i, dim] = AREA_SIZE/2 - DYNAMIC_OBSTACLE_RADIUS
                bird_velocities[i, dim] *= -1
    for i in range(NUM_WILDLIFE):
        wildlife[i] += wildlife_velocities[i]
        for dim in range(2):
            if wildlife[i, dim] < -AREA_SIZE/2 + WILDLIFE_RADIUS:
                wildlife[i, dim] = -AREA_SIZE/2 + WILDLIFE_RADIUS
                wildlife_velocities[i, dim] *= -1
            elif wildlife[i, dim] > AREA_SIZE/2 - WILDLIFE_RADIUS:
                wildlife[i, dim] = AREA_SIZE/2 - WILDLIFE_RADIUS
                wildlife_velocities[i, dim] *= -1
    for i in range(NUM_BIKERS):
        bikers[i] += biker_velocities[i]
        for dim in range(2):
            if bikers[i, dim] < -AREA_SIZE/2 + BIKER_RADIUS:
                bikers[i, dim] = -AREA_SIZE/2 + BIKER_RADIUS
                biker_velocities[i, dim] *= -1
            elif bikers[i, dim] > AREA_SIZE/2 - BIKER_RADIUS:
                bikers[i, dim] = AREA_SIZE/2 - BIKER_RADIUS
                biker_velocities[i, dim] *= -1

def draw_pattern_visualization(ax, t, formation_type):
    if formation_type == "circle":
        circle = patches.Circle((0, 0), CIRCLE_RADIUS, fill=False, color='#00FF00', linestyle='--', linewidth=3, alpha=0.7)
        ax.add_patch(circle)
        angle = t * FORMATION_SPEED
        arrow_start = np.array([CIRCLE_RADIUS * np.cos(angle), CIRCLE_RADIUS * np.sin(angle)])
        arrow_dir = np.array([-np.sin(angle), np.cos(angle)]) * 0.8
        ax.arrow(arrow_start[0], arrow_start[1], arrow_dir[0], arrow_dir[1], head_width=0.3, head_length=0.3, fc='#00FF00', ec='#00FF00', alpha=0.8)
    else:
        t_vals = np.linspace(0, 2*np.pi, 300)
        x_vals, y_vals = [], []
        for t_val in t_vals:
            cos_t = np.cos(t_val)
            sin_t = np.sin(t_val)
            denom = 1 + sin_t**2
            x = FIGURE8_WIDTH * cos_t / denom
            y = FIGURE8_HEIGHT * sin_t * cos_t / denom
            x_vals.append(x)
            y_vals.append(y)
        ax.plot(x_vals, y_vals, '#00FF00', linestyle='--', linewidth=3, alpha=0.7)
        current_t = t * FORMATION_SPEED
        cos_curr = np.cos(current_t)
        sin_curr = np.sin(current_t)
        denom_curr = 1 + sin_curr**2
        curr_x = FIGURE8_WIDTH * cos_curr / denom_curr
        curr_y = FIGURE8_HEIGHT * sin_curr * cos_curr / denom_curr
        ax.scatter(curr_x, curr_y, c='#00FF00', s=100, marker='>', alpha=0.8, zorder=10)

def get_distance_matrix(positions):
    n = len(positions)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            distances[i, j] = distances[j, i] = dist
    return distances

def aodv_routing(source, destination, connectivity_matrix):
    if source == destination:
        return [source]
    visited = [False] * NUM_DRONES
    parent = [-1] * NUM_DRONES
    queue = Queue()
    queue.put(source)
    visited[source] = True
    while not queue.empty():
        current = queue.get()
        if current == destination:
            path = []
            node = destination
            while node != -1:
                path.append(node)
                node = parent[node]
            return path[::-1]
        for neighbor in range(NUM_DRONES):
            if connectivity_matrix[current, neighbor] and not visited[neighbor]:
                visited[neighbor] = True
                parent[neighbor] = current
                queue.put(neighbor)
    return []

def draw_information_panel(ax, step, formation_type):
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 24)
    ax.axis('off')
    title = f'FANET {formation_type.capitalize()} Formation (Km Scale)'
    ax.text(5, 23.5, title, fontsize=12, fontweight='bold', ha='center', bbox=dict(boxstyle="round,pad=0.3", fc='#ADD8E6'))
    ax.text(0.5, 22.5, f'Current Phase: {formation_type.capitalize()} Formation', fontweight='bold', fontsize=8)
    ax.text(0.5, 22.0, f'Step: {step}', fontsize=7, bbox=dict(boxstyle="round,pad=0.2", fc='#FFFF00', alpha=0.7))
    progress = min(100, (step / SIMULATION_STEPS) * 100)
    ax.text(0.5, 21.3, f'Progress: {progress:.1f}%', fontsize=7)
    bar_width, bar_height = 8, 0.3
    ax.add_patch(patches.Rectangle((1, 20.8), bar_width, bar_height, fc='#D3D3D3', ec='#000000'))
    ax.add_patch(patches.Rectangle((1, 20.8), bar_width * progress/100, bar_height, fc='#008000', alpha=0.7))
    distances = get_distance_matrix(positions)
    connectivity = distances < COMM_RANGE
    connected_pairs = np.sum(connectivity) // 2 - NUM_DRONES // 2
    ax.text(0.5, 20.0, 'Network Statistics:', fontweight='bold', fontsize=8)
    ax.text(0.5, 19.5, f'Active Connections: {connected_pairs}', fontsize=7)
    ax.text(0.5, 19.1, f'Network Density: {connected_pairs/(NUM_DRONES*(NUM_DRONES-1)/2)*100:.1f}%', fontsize=7)
    ax.text(0.5, 18.3, 'Formation Parameters:', fontweight='bold', fontsize=8)
    if formation_type == "circle":
        ax.text(0.5, 17.9, f'Circle Radius: {CIRCLE_RADIUS:.1f}km', fontsize=7)
    else:
        ax.text(0.5, 17.9, f'Figure-8 Size: {FIGURE8_WIDTH:.1f}×{FIGURE8_HEIGHT:.1f}km', fontsize=7)
        
    ax.text(0.5, 17.5, f'Formation Speed: {FORMATION_SPEED:.3f}', fontsize=7)
    ax.text(0.5, 17.1, f'Comm Range: {COMM_RANGE:.1f}km', fontsize=7)
    ax.text(0.5, 16.4, 'Camera Detections (Wildlife & Bikers):', fontweight='bold', fontsize=8)
    detection_info = []
    for i in range(NUM_CAMERAS):
        cam_pos = positions[i, :2]
        detected = []
        for j, wild in enumerate(wildlife):
            distance = np.linalg.norm(wild - cam_pos)
            if distance < CAMERA_RADIUS:
                detected.append(f'{wildlife_names[j]} (A{j})')
        for j, biker in enumerate(bikers):
            distance = np.linalg.norm(biker - cam_pos)
            if distance < CAMERA_RADIUS:
                detected.append(f'Biker (BI{j})')
        detection_info.append(f'C{i+1} (D{i}): {", ".join(detected) if detected else "None"}')
    for idx, info in enumerate(detection_info):
        ax.text(0.5, 16.0 - idx * 0.3, info, fontsize=5)
    start_y = 16.0 - len(detection_info) * 0.3
    ax.text(0.5, start_y - 0.3, 'AODV Routing (All Valid Routes):', fontweight='bold', fontsize=8)
    sample_routes = []
    for i in range(NUM_DRONES):
        for j in range(i + 1, NUM_DRONES):
            route = aodv_routing(i, j, connectivity)
            if route and len(route) > 1:
                route_str = ' → '.join([f'D{k}' for k in route])
                sample_routes.append(f'D{i} to D{j}: {route_str}')
    ax.text(0.5, start_y - 0.6, f'Total Routes: {len(sample_routes)}', fontsize=5)
    for idx, route_info in enumerate(sample_routes):
        y_pos = start_y - 0.9 - idx * 0.25
        if y_pos > 0:
            ax.text(0.5, y_pos, route_info, fontsize=4)

def draw_altitude_panel(ax):
    ax.clear()
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.text(2.5, 9.5, 'Altitude Info (km)', fontsize=10, fontweight='bold', ha='center', bbox=dict(boxstyle="round,pad=0.3", fc='#90EE90'))
    for i in range(NUM_DRONES):
        ax.text(2.5, 8.5 - i * 0.8, f'D{i}: {positions[i, 2]:.1f}km', fontsize=8, ha='center', va='center')

def draw_main_scene(formation_type):
    ax_main.clear()
    ax_main.set_xlim(-AREA_SIZE/2 - 1, AREA_SIZE/2 + 1)
    ax_main.set_ylim(-AREA_SIZE/2 - 1, AREA_SIZE/2 + 1)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_aspect('equal')
    ax_main.set_title(f'FANET {formation_type.capitalize()} Formation (Km Scale)', fontsize=12, fontweight='bold')
    ax_main.set_xlabel('Distance (km)', fontsize=10)
    ax_main.set_ylabel('Distance (km)', fontsize=10)
    draw_pattern_visualization(ax_main, formation_time, formation_type)
    
    obstacle_image = load_obstacle_icon()
    if obstacle_image is not None:
        for i, (obstacle, radius) in enumerate(zip(OBSTACLES, OBSTACLE_RADII)):
            try:
                scaled_obstacle = OffsetImage(obstacle_image, zoom=0.1 * (radius / OBSTACLE_RADIUS))
                ab = AnnotationBbox(scaled_obstacle, obstacle, frameon=False)
                ax_main.add_artist(ab)
                ax_main.text(obstacle[0], obstacle[1] + radius + 0.2, f'O{i+1}', ha='center', va='bottom', fontsize=6, color='#000000', fontweight='bold')
            except Exception as e:
                print(f"Error rendering obstacle icon for O{i+1}: {e}")
                circle = patches.Circle(obstacle, radius, color='#FF0000', alpha=0.8)
                ax_main.add_patch(circle)
                ax_main.text(obstacle[0], obstacle[1], f'O{i+1}', ha='center', va='center', fontsize=6, color='#FFFFFF', fontweight='bold')
    else:
        for i, (obstacle, radius) in enumerate(zip(OBSTACLES, OBSTACLE_RADII)):
            circle = patches.Circle(obstacle, radius, color='#FF0000', alpha=0.8)
            ax_main.add_patch(circle)
            ax_main.text(obstacle[0], obstacle[1], f'O{i+1}', ha='center', va='center', fontsize=6, color='#FFFFFF', fontweight='bold')
    
    for i in range(NUM_CAMERAS):
        cam_pos = positions[i, :2]
        circle = patches.Circle(cam_pos, CAMERA_RADIUS, color='#6495ED', alpha=0.4)
        ax_main.add_patch(circle)
        ax_main.text(cam_pos[0], cam_pos[1] + CAMERA_RADIUS + 0.2, f'C{i+1}', ha='center', va='bottom', fontsize=6, color='#6495ED', fontweight='bold')
    
    bird_image = load_bird_icon()
    if bird_image is not None:
        for i, obstacle in enumerate(birds):
            pulse_factor = 1 + 0.1 * np.sin(formation_time * 0.1 + i)
            try:
                scaled_bird = OffsetImage(bird_image, zoom=0.08 * pulse_factor)
                ab = AnnotationBbox(scaled_bird, obstacle, frameon=False)
                ax_main.add_artist(ab)
                ax_main.text(obstacle[0], obstacle[1] + DYNAMIC_OBSTACLE_RADIUS * pulse_factor + 0.2, f'B{i+1}', ha='center', va='bottom', fontsize=6, color='#000000', fontweight='bold')
                ax_main.arrow(obstacle[0], obstacle[1], bird_velocities[i][0] * 15, bird_velocities[i][1] * 15, head_width=0.3, head_length=0.3, fc='#FFFF00', ec='#FFA500', alpha=0.8)
            except Exception as e:
                print(f"Error rendering bird icon for B{i+1}: {e}")
                circle = patches.Circle(obstacle, DYNAMIC_OBSTACLE_RADIUS * pulse_factor, color='#800080', alpha=0.7)
                ax_main.add_patch(circle)
                ax_main.text(obstacle[0], obstacle[1], f'B{i+1}', ha='center', va='center', fontsize=6, color='#FFFFFF', fontweight='bold')
                ax_main.arrow(obstacle[0], obstacle[1], bird_velocities[i][0] * 15, bird_velocities[i][1] * 15, head_width=0.3, head_length=0.3, fc='#FFFF00', ec='#FFA500', alpha=0.8)
    else:
        for i, obstacle in enumerate(birds):
            pulse_factor = 1 + 0.1 * np.sin(formation_time * 0.1 + i)
            circle = patches.Circle(obstacle, DYNAMIC_OBSTACLE_RADIUS * pulse_factor, color='#800080', alpha=0.7)
            ax_main.add_patch(circle)
            ax_main.text(obstacle[0], obstacle[1], f'B{i+1}', ha='center', va='center', fontsize=6, color='#FFFFFF', fontweight='bold')
            ax_main.arrow(obstacle[0], obstacle[1], bird_velocities[i][0] * 15, bird_velocities[i][1] * 15, head_width=0.3, head_length=0.3, fc='#FFFF00', ec='#FFA500', alpha=0.8)
    
    for i, wild in enumerate(wildlife):
        wildlife_image = load_animal_icon(i)
        pulse_factor = 1 + 0.05 * np.sin(formation_time * 0.1 + i)
        if wildlife_image is not None:
            try:
                ab = AnnotationBbox(wildlife_image, wild, frameon=False)
                ax_main.add_artist(ab)
                ax_main.text(wild[0], wild[1] + WILDLIFE_RADIUS * pulse_factor + 0.3, f'A{i}\n{wildlife_names[i]}', ha='center', va='bottom', fontsize=5, color='#000000', fontweight='bold')
            except Exception as e:
                print(f"Error rendering wildlife icon for {wildlife_names[i]} (A{i}): {e}")
                circle = patches.Circle(wild, WILDLIFE_RADIUS * pulse_factor, color='#A52A2A', alpha=0.7)
                ax_main.add_patch(circle)
                ax_main.text(wild[0], wild[1], f'A{i}\n{wildlife_names[i]}', ha='center', va='center', fontsize=5, color='#FFFFFF', fontweight='bold')
        else:
            circle = patches.Circle(wild, WILDLIFE_RADIUS * pulse_factor, color='#A52A2A', alpha=0.7)
            ax_main.add_patch(circle)
            ax_main.text(wild[0], wild[1], f'A{i}\n{wildlife_names[i]}', ha='center', va='center', fontsize=5, color='#FFFFFF', fontweight='bold')
        ax_main.arrow(wild[0], wild[1], wildlife_velocities[i][0] * 15, wildlife_velocities[i][1] * 15, head_width=0.2, head_length=0.2, fc='#A52A2A', ec='#654321', alpha=0.8)
    
    biker_image = load_biker_icon()
    if biker_image is not None:
        for i, biker in enumerate(bikers):
            pulse_factor = 1 + 0.05 * np.sin(formation_time * 0.2 + i)
            try:
                ab = AnnotationBbox(biker_image, biker, frameon=False)
                ax_main.add_artist(ab)
                ax_main.text(biker[0], biker[1] + BIKER_RADIUS * pulse_factor + 0.3, f'BI{i}', ha='center', va='bottom', fontsize=6, color='#000000', fontweight='bold')
                ax_main.arrow(biker[0], biker[1], biker_velocities[i][0] * 15, biker_velocities[i][1] * 15, head_width=0.2, head_length=0.2, fc='#008000', ec='#006400', alpha=0.8)
            except Exception as e:
                print(f"Error rendering biker icon for BI{i}: {e}")
                circle = patches.Circle(biker, BIKER_RADIUS * pulse_factor, color='#008000', alpha=0.7)
                ax_main.add_patch(circle)
                ax_main.text(biker[0], biker[1], f'BI{i}', ha='center', va='center', fontsize=6, color='#FFFFFF', fontweight='bold')
                ax_main.arrow(biker[0], biker[1], biker_velocities[i][0] * 15, biker_velocities[i][1] * 15, head_width=0.2, head_length=0.2, fc='#008000', ec='#006400', alpha=0.8)
    else:
        for i, biker in enumerate(bikers):
            pulse_factor = 1 + 0.05 * np.sin(formation_time * 0.2 + i)
            circle = patches.Circle(biker, BIKER_RADIUS * pulse_factor, color='#008000', alpha=0.7)
            ax_main.add_patch(circle)
            ax_main.text(biker[0], biker[1], f'BI{i}', ha='center', va='center', fontsize=6, color='#FFFFFF', fontweight='bold')
            ax_main.arrow(biker[0], biker[1], biker_velocities[i][0] * 15, biker_velocities[i][1] * 15, head_width=0.2, head_length=0.2, fc='#008000', ec='#006400', alpha=0.8)
    
    for i, target in enumerate(target_positions):
        ax_main.scatter(target[0], target[1], c='#90EE90', s=80, marker='x', linewidth=3, alpha=0.8)
    
    drone_icon = load_drone_icon()
    drone_colors = plt.cm.tab10(np.linspace(0, 1, NUM_DRONES))
    for i, pos in enumerate(positions):
        if drone_icon is not None:
            try:
                ab = AnnotationBbox(drone_icon, pos[:2], frameon=False)
                ax_main.add_artist(ab)
            except Exception as e:
                print(f"Error rendering drone icon for D{i}: {e}")
                ax_main.scatter(pos[0], pos[1], c=[drone_colors[i]], s=500, marker='o', edgecolor='#000000', linewidth=2)
        else:
            ax_main.scatter(pos[0], pos[1], c=[drone_colors[i]], s=500, marker='o', edgecolor='#000000', linewidth=2)
        ax_main.text(pos[0], pos[1]+0.9, f'D{i}', fontsize=8, ha='center', bbox=dict(boxstyle="round,pad=0.2", facecolor=drone_colors[i], alpha=0.8, edgecolor='#000000'))
        vel_magnitude = np.linalg.norm(velocities[i])
        if vel_magnitude > 0.05:
            ax_main.arrow(pos[0], pos[1], velocities[i][0]*4, velocities[i][1]*4, head_width=0.25, head_length=0.25, fc='#FFA500', ec='#FF8C00', alpha=0.8, linewidth=2)
        print(f"Step {step}, Drone {i} Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], Target: [{target_positions[i][0]:.2f}, {target_positions[i][1]:.2f}, {target_positions[i][2]:.2f}]")
    
    distances = get_distance_matrix(positions)
    connectivity = distances < COMM_RANGE
    for i in range(NUM_DRONES):
        for j in range(i+1, NUM_DRONES):
            if connectivity[i, j]:
                ax_main.plot([positions[i][0], positions[j][0]], [positions[i][1], positions[j][1]], '#00FFFF', alpha=0.5, linewidth=1.5)
    
    for i in range(NUM_DRONES):
        for j in range(i + 1, NUM_DRONES):
            route = aodv_routing(i, j, connectivity)
            if len(route) > 1:
                for k in range(len(route)-1):
                    p1, p2 = positions[route[k]], positions[route[k+1]]
                    ax_main.plot([p1[0], p2[0]], [p1[1], p2[1]], '#FF0000', linewidth=2, alpha=0.6, zorder=10)

print("Combined Drone Formation Simulation (Kilometer Scale)")
print("=" * 60)
print("Icon Directory Check:")
print(f"Base Icon Directory: {ICON_DIR}")
print(f"Animals Directory: {os.path.join(ICON_DIR, 'animals')}")
print("=" * 60)
print(f"Simulation area: {AREA_SIZE}km × {AREA_SIZE}km")
print(f"Communication range: {COMM_RANGE}km")
print(f"Circle formation radius: {CIRCLE_RADIUS}km")
print(f"Figure-8 dimensions: {FIGURE8_WIDTH}km × {FIGURE8_HEIGHT}km")
print(f"Cameras: {NUM_CAMERAS}, Wildlife: {NUM_WILDLIFE}, Bikers: {NUM_BIKERS}")
print(f"Obstacles: {len(OBSTACLES)}")
print("=" * 60)

for step in range(SIMULATION_STEPS):
    # Split the simulation in half: first circle, then figure-8
    if step < SIMULATION_STEPS // 2:
        current_formation = "circle"
        target_positions = formation_controller.get_circle_pattern(formation_time)
    else:
        current_formation = "figure-8"
        target_positions = formation_controller.get_figure8_pattern(formation_time)

    for i in range(NUM_DRONES):
        safe_target = enhanced_obstacle_avoidance(positions[i], target_positions[i], OBSTACLES, OBSTACLE_RADII, birds, DYNAMIC_OBSTACLE_RADIUS, wildlife, WILDLIFE_RADIUS, bikers, BIKER_RADIUS)
        new_pos, new_vel = advanced_movement_control(positions[i], velocities[i], safe_target)
        positions[i] = new_pos
        velocities[i] = new_vel
        for dim in range(2):
            if positions[i, dim] < -AREA_SIZE/2 + 0.5:
                positions[i, dim] = -AREA_SIZE/2 + 0.5
                velocities[i, dim] = max(0, velocities[i, dim])
            elif positions[i, dim] > AREA_SIZE/2 - 0.5:
                positions[i, dim] = AREA_SIZE/2 - 0.5
                velocities[i, dim] = min(0, velocities[i, dim])
    
    update_dynamic_obstacles()
    formation_time += 1
    
    draw_main_scene(current_formation)
    draw_information_panel(ax_info, step, current_formation)
    draw_altitude_panel(ax_altitude)
    plt.pause(0.04)

print("✅ Combined Formation Simulation Complete!")
plt.ioff()
plt.show()