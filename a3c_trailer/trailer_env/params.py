# parameters for setup of environment
screen_size = (128, 128)
car_size = (5, 10)
obstacle_size = (10,10)
goal_size = (20, 20)
num_obstacles = 10


# parameters for mrp
reward_goal = 10
reward_collision = -10
reward_timestep = -0.1

# parameters for simulation
dT = 0.5
min_speed = -5
max_speed = 10