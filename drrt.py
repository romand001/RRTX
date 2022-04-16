import os
import sys
import time
import math
import heapq
from collections import deque
from collections.abc import Sequence
import kdtree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import env, plotting, utils, queue

class Node(Sequence):
    # inherits from Sequence to support indexing and thus kd-tree support
    def __init__(self, n):
        self.n = n # make iterable for kd-tree insertion
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.children = set([])
        self.cost_to_goal = np.inf

    def __eq__(self, other):
        return math.hypot(self.x - other.x, self.y - other.y) < 1e-6

    def __lt__(self, other):
        # this is just in case their keys are the same, so a ValueError is not thrown
        return 1

    def __hash__(self):
        # this is required for storing Nodes to sets
        return hash(self.n)

    def __getitem__(self, i):
        # this is required for kd-tree insertion
        return self.n[i]
    
    def __len__(self):
        # this is required for kd-tree insertion
        return 2
   
    def set_parent(self, new_parent):
        # if a parent exists already
        if self.parent:
            self.parent.children.remove(self)
        self.parent = new_parent
        new_parent.children.add(self)

    def distance(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)


class DRRT:

    def __init__(self, x_start, x_goal, robot_radius, step_len, move_dist, 
                 bot_sample_rate, starting_nodes, node_limit=3000, multi_robot=False, 
                 iter_max=10_000, plot_params=None):
        self.s_start = Node(x_start)
        self.s_goal = Node(x_goal)
        self.s_bot = self.s_start
        self.robot_radius = robot_radius
        self.step_len = step_len
        self.move_dist = move_dist
        self.bot_sample_rate = bot_sample_rate
        self.starting_nodes = starting_nodes
        self.node_limit = node_limit
        self.plot_params = plot_params
        self.kd_tree = kdtree.create([self.s_goal])
        sys.setrecursionlimit(3000) # for the kd-tree cus it searches recursively
        self.all_nodes_coor = []
        self.tree_nodes = set([self.s_goal])
        self.robot_position = [self.s_bot.x, self.s_bot.y]
        self.robot_speed = 1.0 # m/s
        self.path = []
        self.other_robots = [] # list of DRRT objects
        self.other_robot_obstacles = [] # list of circular obstacles (x, y, r)

        self.env = env.Env()
        self.plotting = plotting.Plotting(x_start, x_goal)
        self.utils = utils.Utils()

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary
        self.obs_robot = []

        self.started = False
        self.path_to_goal = False
        self.reached_goal = False

        # single robot stuff
        if not multi_robot:
            # plotting
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            self.fig.suptitle('DRRT')
            self.ax.set_xlim(self.env.x_range[0], self.env.x_range[1]+1)
            self.ax.set_ylim(self.env.y_range[0], self.env.y_range[1]+1)
            self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.nodes_scatter = self.ax.scatter([], [], s=4, c='gray', alpha=0.5)
            self.edge_col = LineCollection([], colors='blue', linewidths=0.5)
            self.path_col = LineCollection([], colors='red', linewidths=1.0)
            self.ax.add_collection(self.edge_col)
            self.ax.add_collection(self.path_col)

            # other
            self.iter_max = iter_max

    def step(self):
        # if we reached the goal, just return
        if self.reached_goal:
            return

        # check if planning time is over
        if not self.started and len(self.tree_nodes) > self.starting_nodes:
            self.started = True
        
        # if planning time is over and we have a path, we can move the robot
        if self.started and self.path_to_goal:
            # update robot position and maybe the node it is at
            self.s_bot, self.robot_position = self.utils.update_robot_position(
                self.robot_position, self.s_bot, self.robot_speed, self.move_dist
            )

        if self.s_bot.cost_to_goal < np.inf:
            self.path_to_goal = True
            if self.s_bot == self.s_goal:
                self.reached_goal = True

        ''' MAIN ALGORITHM BEGINS '''

        self.update_robot_obstacles(delta=0.5) # update other robots as obstacles of this robot

        # don't add nodes past limit unless there's currently no path
        if len(self.tree_nodes) >= self.node_limit and self.path_to_goal:
            return

        v = self.random_node()
        v_nearest = self.nearest(v)
        v = self.saturate(v_nearest, v)

        if v and not self.utils.is_collision(v_nearest, v):
            self.extend(v, v_nearest)

    def set_other_robots(self, other_robots):
        # set the other robots that this robot should know about, called by multirobot.py
        self.other_robots = other_robots
        self.other_robot_obstacles = []
        for robot in other_robots:
            self.other_robot_obstacles.append([
                    robot.robot_position[0],
                    robot.robot_position[1],
                    robot.robot_radius
            ])

    def extend(self, v, v_nearest):
        # V_near = self.near(v)
        # if not V_near:
        #     V_near.append(v_nearest)

        # self.find_parent(v, V_near)
        self.find_parent(v, self.tree_nodes)
        if not v.parent:
            return
        self.add_node(v)
                
    def update_click_obstacles(self, event):
        if event.button == 1: # add obstacle
            x, y = int(event.xdata), int(event.ydata)
            self.add_new_obstacle([x, y, 2])
        if event.button == 3 : # remove obstacle on right click
            # find which obstacle was clicked
            obs, shape = self.find_obstacle(event.xdata, event.ydata)
            if obs:
                self.remove_obstacle(obs, shape)

    def update_robot_obstacles(self, delta):
        # delta is distance that robot needs to move for obstackes to be updated
        idx_changed = []
        for idx, other in enumerate(self.other_robots):
            # check if this other robot has moved significantly
            if math.hypot(other.robot_position[0] - self.other_robot_obstacles[idx][0],
                          other.robot_position[1] - self.other_robot_obstacles[idx][1]) > delta:
                idx_changed.append(idx)

        # remove obstacles from their old positions
        for idx in idx_changed:
            self.remove_obstacle(self.other_robot_obstacles[idx], 'robot')

        # add obstacles to their new positions
        for idx in idx_changed:
            # first manage obstacle lists
            self.other_robot_obstacles[idx] = [
                self.other_robots[idx].robot_position[0], 
                self.other_robots[idx].robot_position[1], 
                self.other_robots[idx].robot_radius
            ]
            self.add_new_obstacle(self.other_robot_obstacles[idx], robot=True)

    def add_new_obstacle(self, obs, robot=False):
        x, y, r = obs
        if not robot:
            print("Add circle obstacle at: s =", x, ",", "y =", y)
            self.obs_circle.append(obs)
            self.plotting.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle) # for plotting obstacles
            self.utils.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle) # for collision checking
        else:
            self.obs_robot.append(obs)

        # get possible affected edges to check for collision with new obstacle
        nearby_nodes = self.find_nodes_in_range((x, y), r + self.step_len)
        E = [(u, v) for u in nearby_nodes for v in u.children
             if self.utils.is_intersect_circle(*self.utils.get_ray(u, v), obs[:2], obs[2])]

        # remove children from tree recursively
        queue = deque(list(zip(*E))[1])
        while queue:
            node = queue.pop()
            for child in node.children:
                queue.append(child)
            self.tree_nodes.remove(node)
            self.kd_tree.remove(node)
                
    def remove_obstacle(self, obs, shape):
        if self.obs_robot:
            self.obs_robot.pop()

        # remove obstacle from list if applicable
        if shape == 'circle':
            self.obs_circle.remove(obs)
            self.plotting.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle) # for plotting obstacles
            self.utils.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle) # for collision checking
        elif shape == 'rectangle':
            self.obs_rectangle.remove(obs)
            self.plotting.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle) # for plotting obstacles
            self.utils.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle) # for collision checking
 
    def add_node(self, node_new):
        self.all_nodes_coor.append(np.array([node_new.x, node_new.y])) # for plotting
        self.tree_nodes.add(node_new)
        self.kd_tree.add(node_new)
        # if new node is at start, then path to goal is found
        if node_new == self.s_bot:
            self.s_bot = node_new
            self.path_to_goal = True
            self.update_path(self.s_bot) # update path to goal for plotting

    def saturate(self, v_nearest, v):
        dist, theta = self.get_distance_and_angle(v_nearest, v)
        dist = min(self.step_len, dist)
        node_new = Node((v_nearest.x + dist * math.cos(theta),
                         v_nearest.y + dist * math.sin(theta)))
        return node_new

    def find_parent(self, v, U):
        U = list(U)
        costs = [math.hypot(v.x - u.x, v.y - u.y) + u.cost_to_goal for u in U]
        if not costs:
            return
        min_idx = int(np.argmin(costs))
        best_u = U[min_idx]
        if not self.utils.is_collision(best_u, v):
            v.set_parent(best_u)
            v.cost_to_go = costs[min_idx]
        else:
            del U[min_idx]
            self.find_parent(v, U)
        
    def random_node(self):
        delta = self.utils.delta

        if not self.path_to_goal and np.random.random() < self.bot_sample_rate:
            return Node((self.s_bot.x, self.s_bot.y))

        return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                     np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

    def near(self, v):
        return self.kd_tree.search_nn_dist((v.x, v.y), self.search_radius)

    def nearest(self, v):
        return self.kd_tree.search_nn((v.x, v.y))[0].data

    def update_path(self, node):
        self.path = []
        while node.parent:
            self.path.append(np.array([[node.x, node.y], [node.parent.x, node.parent.y]]))
            node = node.parent
    
    def find_obstacle(self, a, b):
        for (x, y, r) in self.obs_circle:
            if math.hypot(a - x, b - y) <= r:
                return ([x, y, r], 'circle')

        for (x, y, w, h) in self.obs_rectangle:
            if 0 <= a - (x) <= w + 2 and 0 <= b - (y) <= h :
                return ([x, y, w, h], 'rectangle')

    def find_nodes_in_range(self, pos, r):
        return self.kd_tree.search_nn_dist((pos[0], pos[1]), r)

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    @staticmethod
    def get_distance(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy)

