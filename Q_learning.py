import numpy as np
from  generate_state_space import generate_state_space_with_velocity
from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point
import time

def reset():
    dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.
    w = World(dt, width = 120, height = 120, ppm = 6) # The world is 120 meters by 120 meters. ppm is the pixels per meter.

    # Let's add some sidewalks and RectangleBuildings.
    # A Painting object is a rectangle that the vehicles cannot collide with. So we use them for the sidewalks.
    # A RectangleBuilding object is also static -- it does not move. But as opposed to Painting, it can be collided with.
    # For both of these objects, we give the center point and the size.
    w.add(Painting(Point(71.5, 106.5), Point(97, 27), 'gray80')) # We build a sidewalk.
    w.add(RectangleBuilding(Point(72.5, 107.5), Point(95, 25))) # The RectangleBuilding is then on top of the sidewalk, with some margin.

    # Let's repeat this for 4 different RectangleBuildings.
    w.add(Painting(Point(8.5, 106.5), Point(17, 27), 'gray80'))
    w.add(RectangleBuilding(Point(7.5, 107.5), Point(15, 25)))

    w.add(Painting(Point(8.5, 41), Point(17, 82), 'gray80'))
    w.add(RectangleBuilding(Point(7.5, 40), Point(15, 80)))

    w.add(Painting(Point(71.5, 41), Point(97, 82), 'gray80'))
    w.add(RectangleBuilding(Point(72.5, 40), Point(95, 80)))

    # Let's also add some zebra crossings, because why not.
    w.add(Painting(Point(18, 81), Point(0.5, 2), 'white'))
    w.add(Painting(Point(19, 81), Point(0.5, 2), 'white'))
    w.add(Painting(Point(20, 81), Point(0.5, 2), 'white'))
    w.add(Painting(Point(21, 81), Point(0.5, 2), 'white'))
    w.add(Painting(Point(22, 81), Point(0.5, 2), 'white'))

    # A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
    c1 = Car(Point(20,20), np.pi/2)
    w.add(c1)

    c2 = Car(Point(118,90), np.pi, 'blue')
    c2.velocity = Point(3.0,0) # We can also specify an initial velocity just like this.
    w.add(c2)

    # Pedestrian is almost the same as Car. It is a "circle" object rather than a rectangle.
    p1 = Pedestrian(Point(28,81), np.pi)
    p1.max_speed = 10.0 # We can specify min_speed and max_speed of a Pedestrian (and of a Car). This is 10 m/s, almost Usain Bolt.
    w.add(p1)
    w.render() # This visualizes the world we just constructed.
    return w, c1, c2, p1


# Assuming your state space and action space are discrete for simplicity

# Define the state space
# ppm = 6  # pixels per meter
world_size = (121, 121)  # Size in meters
roads = [
    ((71.5, 106.5), (97, 27)),  # Top horizontal road
    ((8.5, 106.5), (17, 27)),   # Top left vertical road
    ((8.5, 41), (17, 82)),      # Bottom left vertical road
    ((71.5, 41), (97, 82)),     # Bottom horizontal road
]
state_space = generate_state_space_with_velocity(world_size, roads, 0, 5, 0.05)
# print(state_space[90])
num_states = len(state_space)
# print(num_states)

# Define the action space

# Define discretized steering angles
# Assuming -1 is full left, 0 is straight, and 1 is full right, with 10 intermediate levels
# Define discretized acceleration values
steering_levels = 11  # Number of levels including -1, 0, 1
steering_angles = np.linspace(-1, 1, steering_levels)

# Define discretized acceleration values
# Assuming the range is from 0 (stop) to 1 (full acceleration), with 10 intermediate levels
acceleration_levels = 11  # Number of levels including 0, 1
acceleration_values = np.linspace(0, 1, acceleration_levels)

# Create a combined discretized action space
action_space = [(steering, acceleration) for steering in steering_angles for acceleration in acceleration_values]
num_actions = len(action_space)
def round_to_precision(value, precision=0.05):
    return round(round(value / precision) * precision,2)

def get_state(agent):
    # Assuming you have a method to get the state representation of the red car
    return [np.round(agent.x), np.round(agent.y), round_to_precision(agent.speed)]


# Initialize Q-table with zeros
Q = np.zeros((num_states, num_actions))

# Define hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Epsilon for epsilon-greedy action selection
epsilon_min = 0.01
epsilon_decay = 0.995

def calculate_goal(center_x = 71.5, center_y = 106.5, size_x = 97, size_y = 2):
    # Given center and size of the top grey region
    # Calculate the top center position of the grey region
    goal_x = center_x
    goal_y = center_y + size_y / 2
    return (goal_x, goal_y)

def reward_function(world, red_car, goal_y):
    collision_penalty = -1000  # Large negative reward for collision
    goal_reward = 1000  # Large positive reward for reaching the goal
    forward_movement_reward = 10  # Positive reward for moving towards the goal
    time_step_penalty = -1  # Small negative reward for each time step to encourage quicker completion

    # Initialize reward
    reward = 0

    # Check for collision with any agent
    if world.collision_exists(red_car):
        return collision_penalty

    
    # Check if the red car has reached the goal
    if red_car.y >= goal_y  and red_car.x >17 and red_car.x <27:
        return goal_reward
    
    try:
        find_state_index(get_state(red_car))
    except ValueError as e:
        return collision_penalty


    # Reward for moving forward towards the goal
    # Normalize the progress reward based on the distance from the start to the goal
    normalized_progress = ((goal_y - abs(red_car.y - goal_y))/ goal_y) * forward_movement_reward
    reward += normalized_progress
    
    # Penalize each time step to encourage quicker completion
    reward += time_step_penalty
    
    return reward
def find_state_index(state):
    condition = (state_space[:, 0] == state[0]) & \
                (state_space[:, 1] == state[1]) & \
                (state_space[:, 2] == state[2])
    indices = np.where(condition)[0]
    if indices.size > 0:
        state_index = indices[0]
        return state_index
    else:
        # Handle the case where the state is not found
        raise ValueError(f"State {state} not found in state space.")

# Training loop
total_rewards = []
num_episodes = 1000
goal_x, goal_y = calculate_goal() 
for episode in range(num_episodes):
    w, c1, c2, p1 = reset()
    state = [c1.x, c1.y, round_to_precision(c1.speed)]
    # print(state)
    total_reward = 0  # Initialize total reward for the episode
    for k in range(400):
        try:
            if (state not in state_space):
            # If the car is out of bounds, raise an exception
                raise ValueError(f"Car is out of bounds at position {state[0], state[1]}")
            state_index = find_state_index(state)
            # np.where((state_space[:, 0] == state[0]) & 
            #             (state_space[:, 1] == state[1]) & 
            #             (state_space[:, 2] == state[2]))[0][0]
            # print(state_index)
            # Red car (agent) decides whether to explore or exploit
            if np.random.rand() < epsilon:
                action_index = np.random.choice(num_actions)  # Explore
            else:
                action_index = np.argmax(Q[state_index])  # Exploit

            # Apply action to the red car
            action = action_space[action_index]
            c1.set_control(action[0], action[1])

            # Predefined behaviors for pedestrian and red car
            if k == 325:
                c2.set_control(0, 0.8)
            if k == 367:
                c2.set_control(0, 0.1)

            # Update the environment
            w.tick()
            w.render()


            next_state = get_state(c1)
            # print("next_state {}".format(next_state))
            if (next_state not in state_space):
                print(next_state)
                # If the car is out of bounds, raise an exception
                raise ValueError(f"Car is out of bounds at position {next_state[0], next_state[1]}")
            next_state_index  = find_state_index(next_state)
            # np.where((state_space[:, 0] == next_state[0]) & 
            #                        (state_space[:, 1] == next_state[1]) & 
            #                        (state_space[:, 2] == next_state[2]))[0][0]
            reward = reward_function(w, c1,goal_y)
            # print(reward)
            done = (w.collision_exists(c1)) or (c1.y >= goal_y)

            # Q-learning update
            Q[state_index, action_index] = (1 - alpha) * Q[state_index, action_index] + \
                                    alpha * (reward + gamma * np.max(Q[next_state_index,:]) - Q[state_index, action_index])

            # Transition to the next state
            state = next_state
            total_reward += reward

            # End the episode if the goal is reached or a collision occurs
            if done:
                break

        except ValueError as e:
    # This will catch ValueError, which we raise when the car is out of bounds
            print(e)
            break  # Exit the loop and proceed to the next episode
    # except Exception as e:
    #     # This will catch any other exceptions that might occur
    #     print(f"An unexpected error occurred: {e}")
    #     break  # Exit the loop and proceed to the next episode
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    total_rewards.append(total_reward)
    print(f"Episode: {episode}, Total Reward: {total_reward}")
    w.close()

    

# Use the trained Q-table to perform actions
