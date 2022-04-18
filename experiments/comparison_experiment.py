import matplotlib.pyplot as plt
from functools import partial
import multiprocessing
from joblib import Parallel, delayed
import pickle as pkl
import time

from agent_instances import *
import multirobot_helpers as mrh

def run_simulation(exp_idx, agent_getter):
    # get agents
    params, robots = agent_getter()
    for robot in robots:
        robot.set_other_robots([other for other in robots if other != robot])
    
    # set up plotting
    if plot_algos[exp_idx]:
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle(algo_names[exp_idx])
        ax.set_xlim(robots[0].env.x_range[0], robots[0].env.x_range[1]+1)
        ax.set_ylim(robots[0].env.y_range[0], robots[0].env.y_range[1]+1)
        bg = fig.canvas.copy_from_bbox(ax.bbox)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show(block=False)
        plt.pause(0.1)
        fig.canvas.blit(ax.bbox)
        # event handling
        fig.canvas.mpl_connect('button_press_event', partial(
                mrh.update_click_obstacles, robots=robots))
    
    start_time = time.time()

    # simulation iterations
    for iter_idx in range(params['iter_max']):
        # RRTX step for each robot
        for robot in robots:
            robot.step()

        # if all robots reached goal, break and return stats
        finished = True
        for robot in robots:
            if not robot.reached_goal:
                finished = False
                break
        if finished:
            break

        # update plot
        if plot_algos[exp_idx] and robots[0].started and iter_idx % 30 == 0:
            fig.canvas.restore_region(bg)
            mrh.env_plot(ax, robots[0]) # pass in any robot, they all know the environment
            for robot in robots:
                mrh.single_bot_plot(ax, robot)
            fig.canvas.blit(ax.bbox)
            fig.canvas.flush_events()

    # return stats
    return {
        'time': time.time() - start_time,
        'path_lengths': [robot.distance_travelled for robot in robots]
    }


if __name__ == '__main__':

    experiment_settings = {
        'toggles': [
            True, # RRTX
            True, # DRRT
            True, # DRRT*
            True  # Velocity Obstacle
        ],
        'num_sim': [
            5, # RRTX
            5, # DRRT
            5, # DRRT*
            1   # Velocity Obstacle
        ]
    }

    agent_getters = [
        get_rrtx_agents,
        get_drrt_agents,
        get_drrt_star_agents,
        get_vel_obs_agents
    ]

    data = {}
    for algo_name in algo_names:
        data[algo_name] = {
            'time': [],
            'path_lengths': []
        }

    experiment_start_time = time.time()

    # iterate over algorithm type
    for exp_idx in range(len(experiment_settings['toggles'])):

        # check if we are running this algorithm
        if experiment_settings['toggles'][exp_idx]:

            print(f'Running {experiment_settings["num_sim"][exp_idx]} {algo_names[exp_idx]} Simulations')

            out = Parallel(n_jobs=multiprocessing.cpu_count()-2)(delayed(run_simulation)
                (
                    exp_idx,
                    agent_getters[exp_idx]
                ) for _ in range(experiment_settings['num_sim'][exp_idx]))

            # add data to dict
            for result in out:
                data[algo_names[exp_idx]]['time'].append(result['time'])
                data[algo_names[exp_idx]]['path_lengths'].append(result['path_lengths'])

    print(f'Experiment took {time.time() - experiment_start_time} seconds')
    print(f'Saving data to pickle file')
    pkl.dump(data, open(f'experiment_data_{time.strftime("%Y%m%d-%H%M%S")}.pkl', 'wb'))

