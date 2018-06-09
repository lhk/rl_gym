# parameters for setup of environment
screen_size = (156, 156)
car_size = (15, 30)
obstacle_size = (10, 10)
goal_size = (50, 5)
num_obstacles = 2

# parameters for mrp
reward_goal = 2
reward_distance = 0.01
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
