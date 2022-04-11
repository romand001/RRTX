import os
import sys
import math
import heapq
from collections.abc import Sequence
import kdtree
# import functools
# from queue import PriorityQueue
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
        # return (self.x - other.x)**2 + (self.y - other.y)**2 < 1e-6
        return self.get_key() == other.get_key()

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
            if self.parent != u and r < self.distance(u):
                N_r_plus_list.remove(u)
                try:
                    u.N_r_minus.remove(self)
                except KeyError:
                    # print('KeyError in RRTX.cull_neighbors(), skipping remove')
                    pass

        self.N_r_plus = set(N_r_plus_list)

    def update_LMC(self, orphan_nodes, r, epsilon):
        # Algorithm 14
        # pass in orphan nodes from main code, make sure the set is maintained properly
        self.cull_neighbors(r)
        # list of tuples: ( u, d_pi(v,u)+lmc(u) )
        lmcs = [(u, self.distance(u) + u.lmc) for u in (self.all_out_neighbors() - orphan_nodes) if u.parent != self]
        if not lmcs:
            return
        # p_prime, self.lmc = min(lmcs, key=lambda x: x[1]) # pretty sure they forgot to set the LMC in Algorithm 14, but they should have
        p_prime, _ = min(lmcs, key=lambda x: x[1]) # pretty sure they forgot to set the LMC in Algorithm 14, but they should have
        # ^ replace _ with self.lmc to update the lmc as well
        self.set_parent(p_prime) # not sure if we need this or literally just set the parent manually without propagating

    def distance(self, other):
        return np.inf if other in self.infinite_dist_nodes else math.hypot(self.x - other.x, self.y - other.y)


class RRTX:
    def __init__(self, x_start, x_goal, step_len, gamma_FOS, epsilon, 
                 start_sample_rate, iter_max):
        self.s_start = Node(x_start)
        self.s_goal = Node(x_goal, lmc=0.0, cost_to_goal=0.0)
        self.s_bot = self.s_start
        self.step_len = step_len
        self.epsilon = epsilon
        self.start_sample_rate = start_sample_rate
        self.search_radius = 0
        self.iter_max = iter_max
        # self.vertices_coor = [[self.s_goal.x, self.s_goal.y]] # for faster nearest neighbor lookup
        self.kd_tree = kdtree.create([self.s_goal])
        self.tree_nodes = [self.s_goal] # this is V_T in the paper
        self.orphan_nodes = set([]) # this is V_T^C in the paper, i.e., nodes that have been disconnected from tree due to obstacles
        self.Q = [] # priority queue of ComparableNodes
        # self.Q = KeyPriorityQueue(key=cmp_to_key(pq_comparator))

        self.env = env.Env()
        self.plotting = plotting.Plotting(x_start, x_goal)
        self.utils = utils.Utils()

        # plotting
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.suptitle('RRTX')
        self.ax.set_xlim(self.env.x_range[0], self.env.x_range[1]+1)
        self.ax.set_ylim(self.env.y_range[0], self.env.y_range[1]+1)
        self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.edge_col = LineCollection([], colors='blue', linewidths=0.5)
        self.ax.add_collection(self.edge_col)

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        ''' RRTX Stuff '''
        # for gamma computation
        self.d = 2 # dimension of the state space
        self.zeta_d = np.pi # volume of the unit d-ball in the d-dimensional Euclidean space
        self.gamma_FOS = gamma_FOS # factor of safety so that gamma > expression from Theorem 38 of RRT* paper
        self.update_gamma() # initialize gamma

    def planning(self):

        # set seed for reproducibility
        np.random.seed(0)

        # set up event handling
        self.fig.canvas.mpl_connect('button_press_event', self.update_obstacles)

        # animation stuff
        plt.gca().set_aspect('equal', adjustable='box')
        self.edge_col.set_animated(True)
        plt.show(block=False)
        plt.pause(0.1)
        self.ax.draw_artist(self.edge_col)
        self.fig.canvas.blit(self.ax.bbox)

        for i in range(self.iter_max):

            self.search_radius = self.shrinking_ball_radius()
            # print(f'search radius: {self.search_radius}')

            # animate
            if i % 10 == 0:
                # add like this for now
                self.edges = []
                for node in self.tree_nodes:
                    if node.parent:
                        self.edges.append(np.array([[node.parent.x, node.parent.y], [node.x, node.y]]))
                self.edge_col.set_segments(np.array(self.edges))
                self.fig.canvas.restore_region(self.bg)
                self.plotting.plot_env(self.ax)
                self.ax.draw_artist(self.edge_col)
                self.fig.canvas.blit(self.ax.bbox)
                self.fig.canvas.flush_events()

            # update robot position if it's moving
            # UNIMPLEMENTED

            v = self.random_node()
            v_nearest = self.nearest(v)
            v = self.saturate(v_nearest, v)

            if v and not self.utils.is_collision(v_nearest, v):
                self.extend(v, v_nearest)
                if v.parent:
                    self.rewire_neighbours(v)
                    self.reduce_inconsistency()

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
                
    def update_obstacles(self, event):
        # Algorithm 8
        x, y = int(event.xdata), int(event.ydata)
        self.add_new_obstacle([x, y, 2])
        self.propagate_descendants()
        self.verify_queue(self.s_bot)
        self.reduce_inconsistency()

    def add_new_obstacle(self, obs):
        # Algorithm 12
        x, y, r = obs
        print("Add circle obstacle at: s =", x, ",", "y =", y)
        self.obs_circle.append(obs)
        self.plotting.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle) # for plotting obstacles
        self.utils.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle) # for collision checking
        self.update_gamma() # free space volume changed, so gamma must change too

        # get all edges that intersect the new circle obstacle
        # DOES EDGE SET "E" INCLUDE EDGES THAT HAVE BEEN REMOVED PREVIOUSLY??? NOT ACCOUNTING RIGHT NOW
        
        E_O = [(v, u) for v in self.tree_nodes if (u:=v.parent) and self.utils.is_intersect_circle(*self.utils.get_ray(v, u), obs[:2], obs[2])]
        for v, u in E_O:
            v.infinite_dist_nodes.add(u)
            u.infinite_dist_nodes.add(v)
            if v.parent == u: # this check is currently irrelevant since u is always v's parent
                self.verify_orphan(v)
                # should theoretically check if the robot is on this edge now, but we do not

        heapq.heapify(self.Q) # reheapify after removing a bunch of elements and ruining queue

    def verify_orphan(self, v):
        # Algorithm 10
        # if v is in Q, remove it from Q and add it to orphan_nodes
        if v in self.Q:
            self.Q.remove(v)
            self.orphan_nodes.add(v)

    def propagate_descendants(self):
        # Algorithm 9
        # recursively add children of nodes in orphan_nodes to orphan_nodes
        orphan_index = 0
        while orphan_index < len(self.orphan_nodes):
            self.orphan_nodes = self.orphan_nodes + self.orphan_nodes[orphan_index].children
            orphan_index += 1
        
        # give all outgoing edges of nodes in orphan_nodes a cost of infinity and update their priority in Q
        for v in self.orphan_nodes:
            for u in (v.all_out_neighbors() + set([v.parent])) - self.orphan_nodes:
                u.cost_to_goal = np.inf # is this the right way to do it?
                verify_queue(u)
        heapq.heapify(self.Q) # reheapify after keys changed to re-sort queue

        # idk what this is
        for v in self.orphan_nodes:
            self.orphan_nodes.remove(v)
            v.cost_to_goal = np.inf
            v.lmc = np.inf
            if v.parent:
                v.parent.children.remove(v)
            v.parent = None

    def verify_queue(self, v):
        # Algorithm 13
        # this does not do the updating, it is done after all changes are made (in propagate_descendants)
        # if v is in Q, update its cost and position, otherwise just add it
        try:
            self.Q.remove((v.get_key(), v))
        except ValueError:
            pass
        heapq.heappush(self.Q, (v.get_key(), v))

    def reduce_inconsistency(self):
        # Algorithm 5
        while len(self.Q) > 0 and (self.Q[0][0] < self.s_bot.get_key() \
                or self.s_bot.lmc != self.s_bot.cost_to_goal or np.isinf(self.s_bot.cost_to_goal) \
                or self.s_bot in list(zip(*self.Q))[1]):
            v = heapq.heappop(self.Q)[1]
        
            if v.cost_to_goal - v.lmc > self.epsilon:
                v.update_LMC(self.orphan_nodes, self.search_radius, self.epsilon)
                self.rewire_neighbours(v)
            
            v.cost_to_goal = v.lmc

    def add_node(self, node_new):
        self.tree_nodes.append(node_new)
        self.kd_tree.add(node_new)

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
        v.set_parent(best_u)
        v.lmc = costs[min_idx] + best_u.lmc

    def rewire_neighbours(self, v):
        # Algorithm 4
        if v.cost_to_goal - v.lmc > self.epsilon:
            v.cull_neighbors(self.search_radius)
            for u in v.all_in_neighbors() - set([v.parent]):
                if u.lmc > v.distance(u) + v.lmc:
                    u.lmc = v.distance(u) + v.lmc
                    u.set_parent(v)
                    if u.cost_to_goal - u.lmc > self.epsilon:
                        self.verify_queue(u)

    def random_node(self, reached_goal=False):
        delta = self.utils.delta

        if not reached_goal and np.random.random() < self.start_sample_rate:
            return Node((self.s_start.x, self.s_start.y))

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

    def extract_path(self, node_end):
        path = [[self.s_goal.x, self.s_goal.y]]
        node = node_end

        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path
    
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
        step_len=3.0, 
        gamma_FOS = 100.0,
        epsilon=0.05,
        start_sample_rate=0.10,  
        iter_max=10000)
    rrtx.planning()


if __name__ == '__main__':
    main()
