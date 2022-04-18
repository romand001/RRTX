import matplotlib.pyplot as plt
from functools import partial

import sys
sys.path.insert(1, '../')
sys.path.insert(1, '../algorithms')

from velocity_obstacle import Velocity_Obstacle
import multirobot_helpers as mrh

if __name__ == '__main__':
    
    vel_obs_params = {
    
        'iter_max': 100000,
        'robot_radius': 0.5,
        'timestep': 0.02,
        
    }

    top_left = (4, 4)
    top_right = (48, 4)
    bottom_left = (4, 28)
    bottom_right = (48, 28)
    
    r1_start = (top_left[0] + 1, top_left[1] + 1)
    r1_goal = (bottom_right[0] - 1, bottom_right[1] - 1)

    r2_start = (top_right[0] + 1, top_right[1] + 1)
    r2_goal = (bottom_left[0] - 1, bottom_left[1] - 1)

    r3_start = (bottom_left[0] + 1, bottom_left[1] + 1)
    r3_goal = (top_right[0] - 1, top_right[1] - 1)

    r4_start = (bottom_right[0] + 1, bottom_right[1] + 1)
    r4_goal = (top_left[0] - 1, top_left[1] - 1)
    
    r1 = Velocity_Obstacle(
        start = r1_start,
        goal = r1_goal,
        robot_radius=vel_obs_params['robot_radius'],
        timestep=vel_obs_params['timestep'],
        iter_max=vel_obs_params['iter_max'],
        plot_params = {
            'robot': True,
            'goal': True,
            'tree': False,
            'path': False,
            'nodes': False,
            'robot_color': 'red',
        }
    )
    
    r2 = Velocity_Obstacle(
        start = r2_start,
        goal = r2_goal,
        robot_radius=vel_obs_params['robot_radius'],
        timestep=vel_obs_params['timestep'],
        iter_max=vel_obs_params['iter_max'],
        plot_params = {
            'robot': True,
            'goal': True,
            'tree': False,
            'path': False,
            'nodes': False,
            'robot_color': 'green',
        }
    )
    
    r3 = Velocity_Obstacle(
        start = r3_start,
        goal = r3_goal,
        robot_radius=vel_obs_params['robot_radius'],
        timestep=vel_obs_params['timestep'],
        iter_max=vel_obs_params['iter_max'],
        plot_params = {
            'robot': True,
            'goal': True,
            'tree': False,
            'path': False,
            'nodes': False,
            'robot_color': 'blue',
        }
    )
    
    r4 = Velocity_Obstacle(
        start = r4_start,
        goal = r4_goal,
        robot_radius=vel_obs_params['robot_radius'],
        timestep=vel_obs_params['timestep'],
        iter_max=vel_obs_params['iter_max'],
        plot_params = {
            'robot': True,
            'goal': True,
            'tree': False,
            'path': False,
            'nodes': False,
            'robot_color': 'magenta',
        }
    )
    
    robots = [r1, r2, r3, r4]
    for robot in robots:
        robot.set_other_robots([other for other in robots if other != robot])
    
    # plotting stuff
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('Multi-Robot Velocity Obstacle', fontsize=16)
    ax.set_xlim(r1.env.x_range[0], r1.env.x_range[1]+1)
    ax.set_ylim(r1.env.y_range[0], r1.env.y_range[1]+1)
    bg = fig.canvas.copy_from_bbox(ax.bbox)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show(block=False)
    plt.pause(0.1)
    fig.canvas.blit(ax.bbox)
    
    for i in range(vel_obs_params['iter_max']):
        # Velocity Obstacle step for each robot
        for robot in robots:
            robot.step()

        # update plot
        if i % 30 == 0:
            fig.canvas.restore_region(bg)
            mrh.env_plot(ax, robots[0]) # pass in any robot, they all know the environment
            for robot in robots:
                mrh.single_bot_plot(ax, robot, sample_based= False)
            fig.canvas.blit(ax.bbox)
            fig.canvas.flush_events()

    print('\nVelocity Obstacle complete!')