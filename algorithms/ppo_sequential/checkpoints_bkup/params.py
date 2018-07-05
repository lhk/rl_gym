# parameters for setup of environment
R = 400

screen_size = (400, 400)
car_size = (10, 10)
obstacle_size = (10, 10)
obs_x_spread = 200
goal_size = (40, 40)

num_obstacles = 8
distance_rescale = R / 4  # only used in radial environment
x_tolerance = R / 4

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
max_speed = 8
stop_on_border_collision = True

# parameters for car
steering_factor = 1.5
