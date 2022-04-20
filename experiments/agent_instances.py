import sys
sys.path.insert(1, '../')
sys.path.insert(1, '../algorithms')
sys.path.insert(1, 'algorithms')

from rrtx import RRTX
from drrt import DRRT
from drrt_star import DRRTStar
from velocity_obstacle import Velocity_Obstacle

''' ROBOT START AND GOALS '''

top_left_in = (5, 26)
top_left_out = (3, 28)
bottom_right_in = (46, 5)
bottom_right_out = (48, 3)
top_right_in = (top_left_in[0], bottom_right_in[1])
top_right_out = (top_left_out[0], bottom_right_out[1])
bottom_left_in = (bottom_right_in[0], top_left_in[1])
bottom_left_out = (bottom_right_out[0], top_left_out[1])

r1_start = top_left_in
r1_goal = bottom_right_out

r2_start = top_right_in
r2_goal = bottom_left_out

r3_start = bottom_right_in
r3_goal = top_left_out

r4_start = bottom_left_in
r4_goal = top_right_out


plot_rrtx = False
plot_drrt = False
plot_drrt_star = False
plot_vel_obs = False
plot_algos = [
    plot_rrtx,
    plot_drrt,
    plot_drrt_star,
    plot_vel_obs
]

algo_names = [
    'RRTX',
    'DRRT',
    'DRRT*',
    'Velocity Obstacles'
]

def get_rrtx_agents():

    rrtx_params = {
        'iter_max': 10_000,
        'robot_radius': 0.5,
        'step_len': 5.0,
        'move_dist': 0.03, # must be < 0.05 bc that's used in update_robot_position()
        'gamma_FOS': 5.0,
        'epsilon': 0.05,
        'bot_sample_rate': 0.10,
        'starting_nodes': 500,
        'node_limit': 2000, # for each robot. after this, new nodes only added if robot gets orphaned
    }

    rrtx1 = RRTX(
        x_start = r1_start,
        x_goal = r1_goal,
        robot_radius = rrtx_params['robot_radius'],
        step_len = rrtx_params['step_len'],
        move_dist = rrtx_params['move_dist'],
        gamma_FOS = rrtx_params['gamma_FOS'],
        epsilon = rrtx_params['epsilon'],
        bot_sample_rate = rrtx_params['bot_sample_rate'],
        starting_nodes = rrtx_params['starting_nodes'],
        node_limit = rrtx_params['node_limit'],
        multi_robot = True,
        plot_params = {
            'robot': plot_rrtx,
            'goal': plot_rrtx,
            'tree': False,
            'path': plot_rrtx,
            'nodes': False,
            'robot_color': 'blue',
            'tree_color': 'blue',
            'path_color': 'blue',
        }
    )

    rrtx2 = RRTX(
        x_start = r2_start,
        x_goal = r2_goal,
        robot_radius = rrtx_params['robot_radius'],
        step_len = rrtx_params['step_len'],
        move_dist = rrtx_params['move_dist'],
        gamma_FOS = rrtx_params['gamma_FOS'],
        epsilon = rrtx_params['epsilon'],
        bot_sample_rate = rrtx_params['bot_sample_rate'],
        starting_nodes = rrtx_params['starting_nodes'],
        node_limit = rrtx_params['node_limit'],
        multi_robot = True,
        plot_params = {
            'robot': plot_rrtx,
            'goal': plot_rrtx,
            'tree': False,
            'path': plot_rrtx,
            'nodes': False,
            'robot_color': 'green',
            'tree_color': 'green',
            'path_color': 'green',
        }
    )

    rrtx3 = RRTX(
        x_start = r3_start,
        x_goal = r3_goal,
        robot_radius = rrtx_params['robot_radius'],
        step_len = rrtx_params['step_len'],
        move_dist = rrtx_params['move_dist'],
        gamma_FOS = rrtx_params['gamma_FOS'],
        epsilon = rrtx_params['epsilon'],
        bot_sample_rate = rrtx_params['bot_sample_rate'],
        starting_nodes = rrtx_params['starting_nodes'],
        node_limit = rrtx_params['node_limit'],
        multi_robot = True,
        plot_params = {
            'robot': plot_rrtx,
            'goal': plot_rrtx,
            'tree': False,
            'path': plot_rrtx,
            'nodes': False,
            'robot_color': 'orange',
            'tree_color': 'orange',
            'path_color': 'orange',
        }
    )

    rrtx4 = RRTX(
        x_start = r4_start,
        x_goal = r4_goal,
        robot_radius = rrtx_params['robot_radius'],
        step_len = rrtx_params['step_len'],
        move_dist = rrtx_params['move_dist'],
        gamma_FOS = rrtx_params['gamma_FOS'],
        epsilon = rrtx_params['epsilon'],
        bot_sample_rate = rrtx_params['bot_sample_rate'],
        starting_nodes = rrtx_params['starting_nodes'],
        node_limit = rrtx_params['node_limit'],
        multi_robot = True,
        plot_params = {
            'robot': plot_rrtx,
            'goal': plot_rrtx,
            'tree': False,
            'path': plot_rrtx,
            'nodes': False,
            'robot_color': 'brown',
            'tree_color': 'brown',
            'path_color': 'brown',
        }
    )

    return rrtx_params, [rrtx1, rrtx2, rrtx3, rrtx4]

def get_drrt_agents():

    drrt_params = {
        'iter_max': 10_000,
        'robot_radius': 0.5,
        'step_len': 3.0,
        'move_dist': 0.01, # must be < 0.05 bc that's used in update_robot_position()
        'gamma_FOS': 5.0,
        'epsilon': 0.05,
        'bot_sample_rate': 0.1,
        'waypoint_sample_rate': 0.5,
        'starting_nodes': 500,
        'node_limit': 5000, # for each robot. after this, new nodes only added if robot gets orphaned
    }

    drrt1 = DRRT(
        x_start = r1_start,
        x_goal = r1_goal,
        robot_radius = drrt_params['robot_radius'],
        step_len = drrt_params['step_len'],
        move_dist = drrt_params['move_dist'],
        bot_sample_rate = drrt_params['bot_sample_rate'],
        waypoint_sample_rate = drrt_params['waypoint_sample_rate'],
        starting_nodes = drrt_params['starting_nodes'],
        node_limit = drrt_params['node_limit'],
        multi_robot = True,
        plot_params = {
            'robot': plot_drrt,
            'goal': plot_drrt,
            'tree': False,
            'path': plot_drrt,
            'nodes': False,
            'robot_color': 'blue',
            'tree_color': 'blue',
            'path_color': 'blue',
        }
    )

    drrt2 = DRRT(
        x_start = r2_start,
        x_goal = r2_goal,
        robot_radius = drrt_params['robot_radius'],
        step_len = drrt_params['step_len'],
        move_dist = drrt_params['move_dist'],
        bot_sample_rate = drrt_params['bot_sample_rate'],
        waypoint_sample_rate = drrt_params['waypoint_sample_rate'],
        starting_nodes = drrt_params['starting_nodes'],
        node_limit = drrt_params['node_limit'],
        multi_robot = True,
        plot_params = {
            'robot': plot_drrt,
            'goal': plot_drrt,
            'tree': False,
            'path': plot_drrt,
            'nodes': False,
            'robot_color': 'green',
            'tree_color': 'green',
            'path_color': 'green',
        }
    )

    drrt3 = DRRT(
        x_start = r3_start,
        x_goal = r3_goal,
        robot_radius = drrt_params['robot_radius'],
        step_len = drrt_params['step_len'],
        move_dist = drrt_params['move_dist'],
        bot_sample_rate = drrt_params['bot_sample_rate'],
        waypoint_sample_rate = drrt_params['waypoint_sample_rate'],
        starting_nodes = drrt_params['starting_nodes'],
        node_limit = drrt_params['node_limit'],
        multi_robot = True,
        plot_params = {
            'robot': plot_drrt,
            'goal': plot_drrt,
            'tree': False,
            'path': plot_drrt,
            'nodes': False,
            'robot_color': 'orange',
            'tree_color': 'orange',
            'path_color': 'orange',
        }
    )

    drrt4 = DRRT(
        x_start = r4_start,
        x_goal = r4_goal,
        robot_radius = drrt_params['robot_radius'],
        step_len = drrt_params['step_len'],
        move_dist = drrt_params['move_dist'],
        bot_sample_rate = drrt_params['bot_sample_rate'],
        waypoint_sample_rate = drrt_params['waypoint_sample_rate'],
        starting_nodes = drrt_params['starting_nodes'],
        node_limit = drrt_params['node_limit'],
        multi_robot = True,
        plot_params = {
            'robot': plot_drrt,
            'goal': plot_drrt,
            'tree': False,
            'path': plot_drrt,
            'nodes': False,
            'robot_color': 'brown',
            'tree_color': 'brown',
            'path_color': 'brown',
        }
    )

    return drrt_params, [drrt1, drrt2, drrt3, drrt4]

def get_drrt_star_agents():

    drrt_star_params = {
        'iter_max': 10_000,
        'robot_radius': 0.5,
        'step_len': 3.0,
        'move_dist': 0.01, # must be < 0.05 bc that's used in update_robot_position()
        'gamma_FOS': 20.0,
        'epsilon': 0.05,
        'bot_sample_rate': 0.1,
        'waypoint_sample_rate': 0.5,
        'starting_nodes': 500,
        'node_limit': 5000, # for each robot. after this, new nodes only added if robot gets orphaned
    }

    drrt_star1 = DRRTStar(
        x_start = r1_start,
        x_goal = r1_goal,
        robot_radius = drrt_star_params['robot_radius'],
        step_len = drrt_star_params['step_len'],
        move_dist = drrt_star_params['move_dist'],
        gamma_FOS = drrt_star_params['gamma_FOS'],
        bot_sample_rate = drrt_star_params['bot_sample_rate'],
        waypoint_sample_rate = drrt_star_params['waypoint_sample_rate'],
        starting_nodes = drrt_star_params['starting_nodes'],
        node_limit = drrt_star_params['node_limit'],
        multi_robot = True,
        plot_params = {
            'robot': plot_drrt_star,
            'goal': plot_drrt_star,
            'tree': False,
            'path': plot_drrt_star,
            'nodes': False,
            'robot_color': 'blue',
            'tree_color': 'blue',
            'path_color': 'blue',
        }
    )

    drrt_star2 = DRRTStar(
        x_start = r2_start,
        x_goal = r2_goal,
        robot_radius = drrt_star_params['robot_radius'],
        step_len = drrt_star_params['step_len'],
        move_dist = drrt_star_params['move_dist'],
        gamma_FOS = drrt_star_params['gamma_FOS'],
        bot_sample_rate = drrt_star_params['bot_sample_rate'],
        waypoint_sample_rate = drrt_star_params['waypoint_sample_rate'],
        starting_nodes = drrt_star_params['starting_nodes'],
        node_limit = drrt_star_params['node_limit'],
        multi_robot = True,
        plot_params = {
            'robot': plot_drrt_star,
            'goal': plot_drrt_star,
            'tree': False,
            'path': plot_drrt_star,
            'nodes': False,
            'robot_color': 'green',
            'tree_color': 'green',
            'path_color': 'green',
        }
    )

    drrt_star3 = DRRTStar(
        x_start = r3_start,
        x_goal = r3_goal,
        robot_radius = drrt_star_params['robot_radius'],
        step_len = drrt_star_params['step_len'],
        move_dist = drrt_star_params['move_dist'],
        gamma_FOS = drrt_star_params['gamma_FOS'],
        bot_sample_rate = drrt_star_params['bot_sample_rate'],
        waypoint_sample_rate = drrt_star_params['waypoint_sample_rate'],
        starting_nodes = drrt_star_params['starting_nodes'],
        node_limit = drrt_star_params['node_limit'],
        multi_robot = True,
        plot_params = {
            'robot': plot_drrt_star,
            'goal': plot_drrt_star,
            'tree': False,
            'path': plot_drrt_star,
            'nodes': False,
            'robot_color': 'orange',
            'tree_color': 'orange',
            'path_color': 'orange',
        }
    )

    drrt_star4 = DRRTStar(
        x_start = r4_start,
        x_goal = r4_goal,
        robot_radius = drrt_star_params['robot_radius'],
        step_len = drrt_star_params['step_len'],
        move_dist = drrt_star_params['move_dist'],
        gamma_FOS = drrt_star_params['gamma_FOS'],
        bot_sample_rate = drrt_star_params['bot_sample_rate'],
        waypoint_sample_rate = drrt_star_params['waypoint_sample_rate'],
        starting_nodes = drrt_star_params['starting_nodes'],
        node_limit = drrt_star_params['node_limit'],
        multi_robot = True,
        plot_params = {
            'robot': plot_drrt_star,
            'goal': plot_drrt_star,
            'tree': False,
            'path': plot_drrt_star,
            'nodes': False,
            'robot_color': 'brown',
            'tree_color': 'brown',
            'path_color': 'brown',
        }
    )

    return drrt_star_params, [drrt_star1, drrt_star2, drrt_star3, drrt_star4]

def get_vel_obs_agents():

    vel_obs_params = {
        'iter_max': 100_000,
        'robot_radius': 0.5,
        'timestep': 0.02,
    }

    vel_obs1 = Velocity_Obstacle(
        start = r1_start,
        goal = r1_goal,
        robot_radius=vel_obs_params['robot_radius'],
        timestep=vel_obs_params['timestep'],
        iter_max=vel_obs_params['iter_max'],
        plot_params = {
            'robot': plot_vel_obs,
            'goal': plot_vel_obs,
            'tree': False,
            'path': False,
            'nodes': False,
            'robot_color': 'blue',
        }
    )

    vel_obs2 = Velocity_Obstacle(
        start = r2_start,
        goal = r2_goal,
        robot_radius=vel_obs_params['robot_radius'],
        timestep=vel_obs_params['timestep'],
        iter_max=vel_obs_params['iter_max'],
        plot_params = {
            'robot': plot_vel_obs,
            'goal': plot_vel_obs,
            'tree': False,
            'path': False,
            'nodes': False,
            'robot_color': 'green',
        }
    )

    vel_obs3 = Velocity_Obstacle(
        start = r3_start,
        goal = r3_goal,
        robot_radius=vel_obs_params['robot_radius'],
        timestep=vel_obs_params['timestep'],
        iter_max=vel_obs_params['iter_max'],
        plot_params = {
            'robot': plot_vel_obs,
            'goal': plot_vel_obs,
            'tree': False,
            'path': False,
            'nodes': False,
            'robot_color': 'orange',
        }
    )

    vel_obs4 = Velocity_Obstacle(
        start = r4_start,
        goal = r4_goal,
        robot_radius=vel_obs_params['robot_radius'],
        timestep=vel_obs_params['timestep'],
        iter_max=vel_obs_params['iter_max'],
        plot_params = {
            'robot': plot_vel_obs,
            'goal': plot_vel_obs,
            'tree': False,
            'path': False,
            'nodes': False,
            'robot_color': 'brown',
        }
    )

    return vel_obs_params, [vel_obs1, vel_obs2, vel_obs3, vel_obs4]

