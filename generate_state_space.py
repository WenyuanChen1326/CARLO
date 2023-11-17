import numpy as np
import matplotlib.pyplot as plt

def generate_state_space(world_size, roads):
    grid_world = np.ones(world_size)

    # Mark the roads in the grid as zero (black)
    for center, size in roads:
        # Calculate the top left and bottom right corners of the road in the grid
        top_left = (int(center[0] - size[0] / 2), int(center[1] - size[1] / 2))
        bottom_right = (int(center[0] + size[0] / 2), int(center[1] + size[1] / 2))
        
        # Mark the road area in the grid as 0 (black)
        # Add boundary checks to prevent marking outside the grid
        # top_left = (max(top_left[0], 0), max(top_left[1], 0))
        bottom_right = (min(bottom_right[0], world_size[0]-1), min(bottom_right[1], world_size[1]-1))
        grid_world[top_left[1]:bottom_right[1]+1, top_left[0]:bottom_right[0]+1] = 0
    state_space = np.argwhere(grid_world.T == 1)
    # print(state_space)
    # # Visualize the state space with a white border
    # fig, ax = plt.subplots(figsize=(12, 12))
    # # Using 'binary' colormap where 0 is black and 1 is white
    # ax.imshow(grid_world, cmap='binary', origin='lower')
    # ax.set_title('State Space Grid World with World Coordinates')
    # plt.xlabel('X Coordinate (meters)')
    # plt.ylabel('Y Coordinate (meters)')
    # # Set the aspect of the plot to be equal
    # ax.set_aspect('equal')
    # # Set the limits of the plot to the limits of the grid
    # ax.set_xlim([0, world_size[0]])
    # ax.set_ylim([0, world_size[1]])
    # plt.show()
    return state_space


def generate_state_space_with_velocity(world_size, roads, min_velocity, max_velocity, velocity_resolution):
    # Generate the initial state space without velocity
    state_space = generate_state_space(world_size, roads)
    
    # Create a discretized velocity space
    velocity_space = np.arange(min_velocity, max_velocity + velocity_resolution, velocity_resolution)
    
    # Expand the state space with velocity
    state_space_expanded = np.array([[x, y, np.round(v,2)] for x, y in state_space for v in velocity_space])
    
    return state_space_expanded





