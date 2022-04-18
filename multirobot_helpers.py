import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

def update_click_obstacles(event, robots):
    # this grabs event from matplotlib and passes it on to all robots
    for robot in robots:
        robot.update_click_obstacles(event)

def env_plot(ax, rrtx):
    # boundary obstacles
    for (ox, oy, w, h) in rrtx.obs_boundary:
        boundary_patch = patches.Rectangle(
            (ox, oy), w, h,
            edgecolor='black',
            facecolor='black',
            fill=True
        )
        ax.add_patch(boundary_patch)
        ax.draw_artist(boundary_patch)

    # rectangle obstacles
    for (ox, oy, w, h) in rrtx.obs_rectangle:
        rect_patch = patches.Rectangle(
            (ox, oy), w, h,
            edgecolor='black',
            facecolor='gray',
            fill=True
        )
        ax.add_patch(rect_patch)
        ax.draw_artist(rect_patch)

    # circle obstacles
    for (ox, oy, r) in rrtx.obs_circle:
        circle_patch = patches.Circle(
            (ox, oy), r,
            edgecolor='black',
            facecolor='gray',
            fill=True
        )
        ax.add_patch(circle_patch)
        ax.draw_artist(circle_patch)

def single_bot_plot(ax, rrtx, sample_based = True):
    params = rrtx.plot_params
    # robot
    if params['robot']:
        bot_patch = patches.Circle(
            (rrtx.robot_position[0], rrtx.robot_position[1]), rrtx.robot_radius,
            edgecolor=params['robot_color'],
            facecolor=params['robot_color'],
            fill=True
        )
        ax.add_patch(bot_patch)
        ax.draw_artist(bot_patch)

    # other robot obstacles (comment out later)
    for (ox, oy, r) in rrtx.obs_robot:
        obs_bot_patch = patches.Circle(
            (ox, oy), r,
            edgecolor='gray',
            facecolor='gray',
            fill=True,
            alpha=0.5
        )
        ax.add_patch(obs_bot_patch)
        ax.draw_artist(obs_bot_patch)

    # goal
    if params['goal']:
        if sample_based:
            goal_patch = patches.Circle(
                (rrtx.s_goal.x, rrtx.s_goal.y), 0.2,
                edgecolor='black',
                facecolor='black',
                fill=True
            )
        else:
            goal_patch = patches.Circle(
                (rrtx.goal[0], rrtx.goal[1]), 0.2,
                edgecolor='black',
                facecolor='black',
                fill=True
            )
        ax.add_patch(goal_patch)
        ax.draw_artist(goal_patch)

    # tree
    if params['tree']:
        edges = []
        for node in rrtx.tree_nodes:
            if node.parent:
                edges.append(np.array([[node.parent.x, node.parent.y], [node.x, node.y]]))

        edge_col = LineCollection([], colors=params['tree_color'], linewidths=0.5)
        ax.add_collection(edge_col)
        edge_col.set_animated(True)
        edge_col.set_segments(np.array(edges))
        ax.draw_artist(edge_col)

    # path
    if params['path']:
        node = rrtx.s_bot
        path = []
        while node.parent:
            path.append(np.array([[node.x, node.y], [node.parent.x, node.parent.y]]))
            node = node.parent

        path_col = LineCollection([], colors=params['path_color'], linewidths=1.0)
        ax.add_collection(path_col)
        path_col.set_animated(True)
        path_col.set_segments(np.array(path))
        ax.draw_artist(path_col)

    # stray nodes
    if params['nodes']:
        if rrtx.all_nodes_coor:
            nodes_scatter = ax.scatter([], [], s=4, c='gray', alpha=0.5)
            nodes_scatter.set_offsets(np.array(rrtx.all_nodes_coor))
            ax.draw_artist(nodes_scatter)

