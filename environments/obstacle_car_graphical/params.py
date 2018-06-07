# parameters for setup of environment
screen_size = (200, 200)
car_size = (20, 40)
obstacle_size = (15, 15)
goal_size = (40, 40)
num_obstacles = 3

# parameters for mrp
reward_goal = 2
reward_collision = -1
reward_timestep = -0.1
timeout = 600

# parameters for simulation
dT = 2
min_speed = -8
max_speed = 10
stop_on_border_collision = True

# parameters for car
steering_factor = 1.5
