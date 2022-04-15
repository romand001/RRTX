from functools import partial

from rrtx import RRTX, Node
from multirobot_helper import *

if __name__ == '__main__':

    rrt_params = {
        'iter_max': 10_000,
        'robot_radius': 0.5,
        'step_len': 3.0,
        'move_dist': 0.01, # must be < 0.05 bc that's used in update_robot_position()
        'gamma_FOS': 10.0,
        'epsilon': 0.05,
        'bot_sample_rate': 0.10,
        'planning_time': 5.0,
    }

    plot_params_1 = {
        'robot': True,
        'goal': True,
        'tree': True,
        'path': True,
        'nodes': True,
        'robot_color': 'purple',
        'tree_color': 'green',
        'path_color': 'red',
    }

    plot_params_2 = {
        'robot': True,
        'goal': True,
        'tree': True,
        'path': True,
        'nodes': True,
        'robot_color': 'brown',
        'tree_color': 'blue',
        'path_color': 'orange',
    }   

    r1_start = (3, 20)
    r1_goal = (20, 20)

    r2_start = (r1_goal[0] + 1, r1_goal[1])
    r2_goal = (r1_start[0] + 1, r1_start[1])

    r1 = RRTX(
        x_start = r1_start,
        x_goal = r1_goal,
        robot_radius = rrt_params['robot_radius'],
        step_len = rrt_params['step_len'],
        move_dist = rrt_params['move_dist'],
        gamma_FOS = rrt_params['gamma_FOS'],
        epsilon = rrt_params['epsilon'],
        bot_sample_rate = rrt_params['bot_sample_rate'],
        planning_time = rrt_params['planning_time'],
        multi_robot = True
    )

    r2 = RRTX(
        x_start = r2_start,
        x_goal = r2_goal,
        robot_radius = rrt_params['robot_radius'],
        step_len = rrt_params['step_len'],
        move_dist = rrt_params['move_dist'],
        gamma_FOS = rrt_params['gamma_FOS'],
        epsilon = rrt_params['epsilon'],
        bot_sample_rate = rrt_params['bot_sample_rate'],
        planning_time = rrt_params['planning_time'],
        multi_robot = True
    )

    r1.set_other_robots([r2])
    r2.set_other_robots([r1])

    robots = [r1, r2]
    plot_params = [plot_params_1, plot_params_2]

    # plotting stuff
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('RRTX')
    ax.set_xlim(r1.env.x_range[0], r1.env.x_range[1]+1)
    ax.set_ylim(r1.env.y_range[0], r1.env.y_range[1]+1)
    bg = fig.canvas.copy_from_bbox(ax.bbox)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show(block=False)
    plt.pause(0.1)
    fig.canvas.blit(ax.bbox)

    # event handling
    fig.canvas.mpl_connect('button_press_event', partial(update_click_obstacles, param=robots))

    for i in range(rrt_params['iter_max']):
        # RRTX step for each robot
        for robot in robots:
            robot.step()

        # update plot
        if robots[0].started and i % 10 == 0:
            fig.canvas.restore_region(bg)
            env_plot(ax, robots[0]) # pass in any robot, they all know the environment
            for robot_idx in range(len(robots)):
                single_bot_plot(ax, robots[robot_idx], plot_params[robot_idx])
            fig.canvas.blit(ax.bbox)
            fig.canvas.flush_events()

