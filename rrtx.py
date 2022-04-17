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
    def __init__(self, n, lmc=np.inf, cost_to_goal=np.inf):
        self.n = n # make iterable for kd-tree insertion
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.children = set([])
        self.cost_to_goal = cost_to_goal
        self.lmc = lmc
        self.infinite_dist_nodes = set([]) # set of nodes u where d_pi(v,u) has been set to infinity after adding an obstacle
        self.N_o_plus = set([]) # outgoing original neighbours
        self.N_o_minus = set([]) # incoming original neighbours
        self.N_r_plus = set([]) # outgoing running in neighbours
        self.N_r_minus = set([]) # incoming running in neighbours

    def __eq__(self, other):
        # this is required for the checking if a node is "in" self.Q, but idk what the condition should be
        return id(self) == id(other) or self.get_key() == other.get_key() or \
            math.hypot(self.x - other.x, self.y - other.y) < 1e-6
        # return self.get_key() == other.get_key()
        # return self.x == other.x and self.y == other.y

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

    def all_out_neighbors(self):
        return self.N_o_plus.union(self.N_r_plus)
    
    def all_in_neighbors(self):
        return self.N_o_minus.union(self.N_r_minus)
   
    def set_parent(self, new_parent):
        # if a parent exists already
        if self.parent:
            try:
                self.parent.children.remove(self)
            except:
                print('KeyError in set parent')
        self.parent = new_parent
        new_parent.children.add(self)

    def get_key(self):
        return (min(self.cost_to_goal, self.lmc), self.cost_to_goal)

    def cull_neighbors(self, r):
        # Algorithm 3
        N_r_plus_list = list(self.N_r_plus) # can't remove from set while iterating over it
        for u in N_r_plus_list:
            # switched order of conditions in if statement to be faster
            if self.parent and self.parent != u and r < self.distance(u):
                N_r_plus_list.remove(u)
                try:
                    u.N_r_minus.remove(self)
                except KeyError:
                    # print('KeyError in RRTX.cull_neighbors(), skipping remove')
                    pass

        self.N_r_plus = set(N_r_plus_list)

    def update_LMC(self, orphan_nodes, r, epsilon, utils):
        # Algorithm 14
        # pass in orphan nodes from main code, make sure the set is maintained properly
        self.cull_neighbors(r)
        # list of tuples: ( u, d_pi(v,u)+lmc(u) )
        lmcs = [(u, self.distance(u) + u.lmc) for u in (self.all_out_neighbors() - orphan_nodes) if u.parent and u.parent != self]
        if not lmcs:
            return
        p_prime, lmc_prime = min(lmcs, key=lambda x: x[1])
        if lmc_prime < self.lmc and not utils.is_collision(self, p_prime): # added collision check, not in pseudocode
            self.lmc = lmc_prime # lmc update is done in Julia code
            self.set_parent(p_prime) # not sure if we need this or literally just set the parent manually without propagating

    def distance(self, other):
        return np.inf if other in self.infinite_dist_nodes else math.hypot(self.x - other.x, self.y - other.y)


class RRTX:

    def __init__(self, x_start, x_goal, robot_radius, step_len, move_dist, gamma_FOS, epsilon, 
                 bot_sample_rate, starting_nodes, node_limit=3000, multi_robot=False,
                 iter_max=10_000, plot_params=None):
        self.s_start = Node(x_start)
        self.s_goal = Node(x_goal, lmc=0.0, cost_to_goal=0.0)
        self.s_bot = self.s_start
        self.robot_radius = robot_radius
        self.step_len = step_len
        self.move_dist = move_dist
        self.epsilon = epsilon
        self.bot_sample_rate = bot_sample_rate
        self.starting_nodes = starting_nodes
        self.node_limit = node_limit
        self.plot_params = plot_params
        self.search_radius = 0.0
        self.kd_tree = kdtree.create([self.s_goal])
        sys.setrecursionlimit(3000) # for the kd-tree cus it searches recursively
        self.all_nodes_coor = []
        self.tree_nodes = [self.s_goal] # this is V_T in the paper
        self.orphan_nodes = set([]) # this is V_T^C in the paper, i.e., nodes that have been disconnected from tree due to obstacles
        self.Q = [] # priority queue of ComparableNodes
        self.robot_position = [self.s_bot.x, self.s_bot.y]
        self.robot_speed = 1.0 # m/s
        self.path = []
        self.other_robots = [] # list of RRTX objects
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

        # for gamma computation
        self.d = 2 # dimension of the state space
        self.zeta_d = np.pi # volume of the unit d-ball in the d-dimensional Euclidean space
        self.gamma_FOS = gamma_FOS # factor of safety so that gamma > expression from Theorem 38 of RRT* paper
        self.update_gamma() # initialize gamma

        self.started = False
        self.path_to_goal = False
        self.reached_goal = False

        # single robot stuff
        if not multi_robot:
            # plotting
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            self.fig.suptitle('RRTX')
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

    def planning(self):

        # set seed for reproducibility
        np.random.seed(0)

        # set up event handling
        self.fig.canvas.mpl_connect('button_press_event', self.update_click_obstacles)

        # animation stuff
        plt.gca().set_aspect('equal', adjustable='box')
        self.edge_col.set_animated(True)
        self.path_col.set_animated(True)
        plt.show(block=False)
        plt.pause(0.1)
        self.ax.draw_artist(self.edge_col)
        self.fig.canvas.blit(self.ax.bbox)
        start_time = time.time()
        prev_plotting = time.time()
        first_time = True

        for i in range(self.iter_max):

            # update robot position
            run_time = time.time() - start_time
            if self.path_to_goal and run_time > 5:
                # timing stuff
                if first_time:
                    prev_time = time.time()
                    first_time = False
                elapsed_time = time.time() - prev_time
                prev_time = time.time()

                # update robot position and maybe the node it is at
                self.s_bot, self.robot_position = self.utils.update_robot_position(
                    self.robot_position, self.s_bot, 
                    self.robot_speed, 0.01)

            # animate
            if run_time > 5 or run_time < 0.01:

                # only update the plot at 5 Hz
                elapsed_plotting = time.time() - prev_plotting
                if elapsed_plotting >= 0.2:
                    prev_plotting = time.time()
                    self.edges = []
                    for node in self.tree_nodes:
                        if node.parent:
                            self.edges.append(np.array([[node.parent.x, node.parent.y], [node.x, node.y]]))

                    self.fig.canvas.restore_region(self.bg)

                    # all nodes
                    self.nodes_scatter.set_offsets(np.array(self.all_nodes_coor))
                    self.ax.draw_artist(self.nodes_scatter)

                    # tree edges
                    self.edge_col.set_segments(np.array(self.edges))
                    self.ax.draw_artist(self.edge_col)

                    # path to goal edges
                    self.path_col.set_segments(np.array(self.path))
                    self.ax.draw_artist(self.path_col)
                    
                    # obstacles and robot
                    self.plotting.plot_env(self.ax)
                    self.plotting.plot_robot(self.ax, self.robot_position, self.robot_radius)

                    self.fig.canvas.blit(self.ax.bbox)
                    self.fig.canvas.flush_events()

            self.search_radius = self.shrinking_ball_radius()

            v = self.random_node()
            v_nearest = self.nearest(v)
            v = self.saturate(v_nearest, v)

            if v and not self.utils.is_collision(v_nearest, v):
                self.extend(v, v_nearest)
                if v.parent:
                    self.rewire_neighbours(v)
                    self.reduce_inconsistency()

            if self.s_bot.cost_to_goal < np.inf:
                self.path_to_goal = True

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

        self.search_radius = self.shrinking_ball_radius()

        self.update_robot_obstacles(delta=0.5) # update other robots as obstacles of this robot

        # don't add nodes past limit unless there's currently no path
        if len(self.tree_nodes) >= self.node_limit and self.path_to_goal:
            return

        v = self.random_node()
        v_nearest = self.nearest(v)
        v = self.saturate(v_nearest, v)

        if v and not self.utils.is_collision(v_nearest, v):
            self.extend(v, v_nearest)
            if v.parent:
                self.rewire_neighbours(v)
                self.reduce_inconsistency()

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
        # Algorithm 2
        V_near = self.near(v)

        ### THIS WAS NOT IN PAPER, BUT IN JULIA CODE
        if not V_near:
            V_near.append(v_nearest)

        self.find_parent(v, V_near)
        if not v.parent:
            return
        self.add_node(v)
        # child has already been added to parent's children in call to set_parent()
        for u in V_near:
            # collisions are symmetric for us
            if not self.utils.is_collision(u, v):
                v.N_o_plus.add(u)
                v.N_o_minus.add(u)
                u.N_r_plus.add(v)
                u.N_r_minus.add(v)
                
    def update_click_obstacles(self, event):
        # Algorithm 8, for obstacles added by clicking
        # Algorithm 8
        if event.button == 1: # add obstacle
            x, y = int(event.xdata), int(event.ydata)
            self.add_new_obstacle([x, y, 2])
            self.propagate_descendants()
            self.verify_queue(self.s_bot)
            heapq.heapify(self.Q)
            self.reduce_inconsistency()
        if event.button == 3 : # remove obstacle on right click
            # find which obstacle was clicked
            obs, shape = self.find_obstacle(event.xdata, event.ydata)
            if obs:
                self.remove_obstacle(obs, shape)
            self.reduce_inconsistency()

    def update_robot_obstacles(self, delta):
        # Algorithm 8, for robot obstacles
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

        self.reduce_inconsistency()

        # add obstacles to their new positions
        for idx in idx_changed:
            # first manage obstacle lists
            self.other_robot_obstacles[idx] = [
                self.other_robots[idx].robot_position[0], 
                self.other_robots[idx].robot_position[1], 
                self.other_robots[idx].robot_radius
            ]
            self.add_new_obstacle(self.other_robot_obstacles[idx], robot=True)

        self.propagate_descendants()
        self.verify_queue(self.s_bot)
        heapq.heapify(self.Q)
        self.reduce_inconsistency()

    def add_new_obstacle(self, obs, robot=False):
        # Algorithm 12
        x, y, r = obs
        if not robot:
            print("Add circle obstacle at: s =", x, ",", "y =", y)
            self.obs_circle.append(obs)
            self.plotting.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle) # for plotting obstacles
            self.utils.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle) # for collision checking
        else:
            self.obs_robot.append(obs)

        self.update_gamma() # free space volume changed, so gamma must change too

        # get all edges that intersect the new circle obstacle
        nearby_nodes = self.find_nodes_in_range((x, y), r + self.step_len)
        E_O = [(v, u) for v in nearby_nodes for u in v.all_out_neighbors() 
               if self.utils.is_intersect_circle(u.n, v.n, (x, y), r)]
        for v, u in E_O:
            v.infinite_dist_nodes.add(u)
            u.infinite_dist_nodes.add(v)
            if v.parent and v.parent == u:
                self.verify_orphan(v)
                # should theoretically check if the robot is on this edge now, but we do not
                # v.parent.children.remove(v) # these two lines are from the Julia code
                # v.parent = None 
                
        heapq.heapify(self.Q) # reheapify after removing a bunch of elements and ruining queue

    def remove_obstacle(self, obs, shape):
        # Algorithm 11
        if self.obs_robot:
            self.obs_robot.pop()
        # Find all nodes going through new region
        if shape == 'circle' or shape == 'robot': 
            nodes_affected = self.find_nodes_in_range(obs[:2], self.utils.delta + self.step_len + obs[2]) # robot size+step length + obstacle size
        elif shape == 'rectangle':
            nodes_affected = self.find_nodes_in_range(obs[:2], self.utils.delta + self.step_len + max(obs[2],obs[3]))
            
        # remove obstacle from list if applicable
        if shape == 'circle':
            self.obs_circle.remove(obs)
            self.plotting.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle) # for plotting obstacles
            self.utils.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle) # for collision checking\
            self.update_gamma() # free space volume changed, so gamma must change too
        elif shape == 'rectangle':
            self.obs_rectangle.remove(obs)
            self.plotting.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle) # for plotting obstacles
            self.utils.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle) # for collision checking\
            self.update_gamma() # free space volume changed, so gamma must change too

        for node in nodes_affected:
            node.update_LMC(self.orphan_nodes, self.search_radius, self.epsilon, self.utils)
            if node.lmc != node.cost_to_goal:
                self.verify_queue(node)
        heapq.heapify(self.Q)
 
    def verify_orphan(self, v):
        # Algorithm 10
        # if v is in Q, remove it from Q and add it to orphan_nodes
        key = self.node_in_queue(v)
        if key is not None:
            self.Q.remove((key, v))
        self.orphan_nodes.add(v)

    def propagate_descendants(self):
        # Algorithm 9
        if not self.orphan_nodes:
            return
        # recursively add children of nodes in orphan_nodes to orphan_nodes using BFS
        orphan_queue = deque(list(self.orphan_nodes))
        while orphan_queue:
            node = orphan_queue.pop()
            for child in node.children:
                orphan_queue.append(child)
                self.orphan_nodes.add(child)

        # check if robot node got orphaned
        if self.s_bot in self.orphan_nodes:
            print('robot node got orphaned')
            self.path_to_goal = False
        
        # put all outgoing neighbours of orphan nodes in Q and tell them to rewire
        for v in self.orphan_nodes:
            for u in (v.all_out_neighbors().union(set([v.parent]))) - self.orphan_nodes:
                u.cost_to_goal = np.inf
                self.verify_queue(u)
        heapq.heapify(self.Q) # reheapify after keys changed to re-sort queue

        # clear orphans, set their costs to infinity, empty their parent
        for v in self.orphan_nodes:
            # self.orphan_nodes.remove(v)
            v.cost_to_goal = np.inf
            v.lmc = np.inf
            if v.parent:
                v.infinite_dist_nodes.add(v.parent)
                v.parent.infinite_dist_nodes.add(v)
                v.parent.children.remove(v)
                v.parent = None
            try:
                self.tree_nodes.remove(v) # NOT IN THE PSEUDOCODE
                self.kd_tree.remove(v)
            except ValueError:
                pass

        self.orphan_nodes = set([]) # reset orphan_nodes to empty set

    def verify_queue(self, v):
        # Algorithm 13
        # this does not do the updating, it is done after all changes are made (in propagate_descendants)
        # if v is in Q, update its cost and position, otherwise just add it
        key = self.node_in_queue(v)
        # if v is already in Q, remove it first before adding it with updated cost
        if key is not None:
            self.Q.remove((key, v))
        heapq.heappush(self.Q, (v.get_key(), v))

    def reduce_inconsistency(self):
        # Algorithm 5
        while len(self.Q) > 0 and (self.Q[0][0] < self.s_bot.get_key() \
                or self.s_bot.lmc != self.s_bot.cost_to_goal or np.isinf(self.s_bot.cost_to_goal) \
                or self.s_bot in list(zip(*self.Q))[1]):

            try:
                v = heapq.heappop(self.Q)[1]
            except TypeError:
                print('something went wrong with the queue')
        
            if v.cost_to_goal - v.lmc > self.epsilon:
                v.update_LMC(self.orphan_nodes, self.search_radius, self.epsilon, self.utils)
                self.rewire_neighbours(v)
            
            v.cost_to_goal = v.lmc

    def add_node(self, node_new):
        self.all_nodes_coor.append(np.array([node_new.x, node_new.y])) # for plotting
        self.tree_nodes.append(node_new)
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
        # Algorithm 6
        # skip collision check because it is done in "near()"
        costs = [math.sqrt((v.x - u.x)**2 + (v.y - u.y)**2) + u.lmc for u in U]
        if not costs:
            return
        min_idx = int(np.argmin(costs))
        best_u = U[min_idx]
        if not self.utils.is_collision(best_u, v):
            v.set_parent(best_u)
            v.lmc = costs[min_idx] + best_u.lmc
        else:
            del U[min_idx]
            self.find_parent(v, U)
        
    def rewire_neighbours(self, v):
        # Algorithm 4
        if v.cost_to_goal - v.lmc > self.epsilon:
            v.cull_neighbors(self.search_radius)
            for u in v.all_in_neighbors() - set([v.parent]):
                if u.lmc > v.distance(u) + v.lmc and not self.utils.is_collision(u, v): # added collision check (Julia)
                    u.lmc = v.distance(u) + v.lmc
                    u.set_parent(v)
                    if u.cost_to_goal - u.lmc > self.epsilon:
                        self.verify_queue(u)
        heapq.heapify(self.Q)

        self.update_path(self.s_bot) # update path to goal for plotting

    def random_node(self):
        delta = self.utils.delta

        if not self.path_to_goal and np.random.random() < self.bot_sample_rate:
            return Node((self.s_bot.x, self.s_bot.y))

        return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                     np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

    def update_gamma(self):
        '''
        computes and updates gamma required for shrinking ball radius
        - gamma depends on the free space volume, so changes when obstacles are added or removed
        - this assumes that obstacles don't overlap
        '''
        mu_X_free = (self.x_range[1] - self.x_range[0]) * (self.y_range[1] - self.y_range[0])
        for (_, _, r) in self.obs_circle:
            mu_X_free -= np.pi * r ** 2
        for (_, _, w, h) in self.obs_rectangle:
            mu_X_free -= w * h

        self.gamma = self.gamma_FOS * (2 * (1 + 1/self.d))**(1/self.d) * (mu_X_free/self.zeta_d)**(1/self.d) # optimality condition from Theorem 38 of RRT* paper

    def shrinking_ball_radius(self):
        '''
        Computes and returns the radius for the shrinking ball
        '''
        return min(self.step_len, self.gamma * np.log(len(self.tree_nodes)+1) / len(self.tree_nodes))

    def near(self, v):
        return self.kd_tree.search_nn_dist((v.x, v.y), self.search_radius)

    def nearest(self, v):
        return self.kd_tree.search_nn((v.x, v.y))[0].data

    def update_path(self, node):
        self.path = []
        while node.parent:
            self.path.append(np.array([[node.x, node.y], [node.parent.x, node.parent.y]]))
            node = node.parent
    
    def node_in_queue(self, node):
        if not self.Q:
            return None
        keys, nodes = list(zip(*self.Q))
        try:
            idx = nodes.index(node)
            return keys[idx]
        except ValueError:
            return None

    def find_obstacle(self, a , b):
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

def main():
    x_start = (18, 8)  # Starting node
    x_goal = (37, 18)  # Goal node


    rrtx = RRTX(
        x_start=x_start, 
        x_goal=x_goal, 
        robot_radius=0.5,
        step_len=3.0, 
        move_dist=0.01,
        gamma_FOS = 100.0,
        epsilon=0.05,
        bot_sample_rate=0.10,
        planning_time=5.0,
        iter_max=10_000)
    rrtx.planning()


if __name__ == '__main__':
    main()
