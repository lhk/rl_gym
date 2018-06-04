# parameters for setup of environment
screen_size = (256, 256)
car_size = (15, 30)
obstacle_size = (15, 15)
goal_size = (40, 40)
num_obstacles = 2

# parameters for mrp
reward_goal = 100
reward_collision = -10
reward_timestep = -0.1
timeout = 600

# parameters for simulation
dT = 1
min_speed = -8
max_speed = 10
stop_on_border_collision= True

# parameters for car
steering_factor = 1.5
