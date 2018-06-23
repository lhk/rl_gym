# parameters for setup of environment
screen_size = (400, 400)
car_size = (20, 40)
obstacle_size = (10, 10)
goal_size = (60, 20)
num_obstacles = 5
distance_rescale = 100 # only used in radial environment

# parameters for mrp
reward_goal = 2.
reward_distance = 0.005
reward_collision = -4.
reward_timestep = -0.1
timeout = 600
max_dist = 1.05

# parameters for simulation
dT = 1.5
min_speed = -8
max_speed = 10
stop_on_border_collision = True

# parameters for car
steering_factor = 1.5
