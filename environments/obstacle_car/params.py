# parameters for setup of environment
screen_size = (256, 256)
car_size = (15, 30)
obstacle_size = (10, 10)
goal_size = (50, 5)
num_obstacles = 2
distance_rescale = 100 # only used in radial environment

# parameters for mrp
reward_goal = 5
reward_distance = 0.01
reward_collision = -5
reward_timestep = -0.1
timeout = 600
max_dist = 1.05

# parameters for simulation
dT = 0.7
min_speed = -8
max_speed = 10
stop_on_border_collision = True

# parameters for car
steering_factor = 1.5
