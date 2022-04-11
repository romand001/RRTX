import os
import sys
import math
import heapq
# import functools
# from queue import PriorityQueue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import env, plotting, utils, queue

class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.children = set([])
        self.cost_to_parent = 0.0
        self.cost_to_goal = 0.0
        self.lmc = 0.0
        self.infinite_dist_nodes = set([]) # set of nodes u where d_pi(v,u) has been set to infinity after adding an obstacle
        # each node has original neighbors, running (new) in/out neighbors, 
        # AND parent, and in-child set. moving through parent/child should always yield shortest path
        self.og_neighbor = set([])
        self.out_neighbor = set([])
        self.in_neighbor = set([])

    def __eq__(self, other):
        # this is required for the checking if a node is "in" self.Q, but idk what the condition should be
        return (self.x - other.x)**2 + (self.y - other.y)**2 < 1e-6

    def all_out_neighbors(self):
        return self.out_neighbor.union(self.og_neighbor)
    
    def all_in_neighbors(self):
        return self.out_neighbor.union(self.og_neighbor)
   
    def set_parent(self, new_parent):
        # if a parent exists already
        if old_parent := self.parent:
            old_parent.children.remove(self)
        self.parent = new_parent
        self.cost_to_parent = math.hypot(self.x - new_parent.x, self.y - new_parent.y)
        self.cost_to_goal = new_parent.cost_to_goal + self.cost_to_parent
        new_parent.children.add(self)
        if old_parent:
            self.update_cost_recursive()

    def update_cost_recursive(self):
        self.cost_to_goal = self.parent.cost_to_goal + self.cost_to_parent
        for child in self.children:
            child.update_cost_recursive()

    def get_key(self):
        return (min(self.cost_to_goal, self.lmc), self.cost_to_goal)

    def cull_neighbors(self, r):
        # Algorithm 3
        for u in self.out_neighbor:
            # switched order of conditions in if statement to be faster
            if self.parent != u and r < self.distance(u):
                self.out_neighbor.remove(u)
                u.in_neighbor.remove(self)

    def update_LMC(self, orphan_nodes, r):
        # Algorithm 14
        # pass in orphan nodes from main code, make sure the set is maintained properly
        self.cull_neighbors(r)
        # list of tuples: ( u, d_pi(v,u)+lmc(u) )
        lmcs = [(u, self.distance(u) + u.lmc) for u in (self.all_out_neighbors() - orphan_nodes) if u.parent != self]
        if not lmcs:
            return
        p_prime, self.lmc = min(lmcs, key=lambda x: x[1]) # pretty sure they forgot to set the LMC in Algorithm 14, but they should have
        self.set_parent(p_prime) # not sure if we need this or literally just set the parent manually without propagating

    def distance(self, other):
        return np.inf if other in self.infinite_dist_nodes else math.hypot(self.x - other.x, self.y - other.y)

# @functools.total_ordering
# class ComparableNode(Node):
#     '''  
#     This is a wrapper class for the Node class to provide a key comparison 
#     for the priority queue
#     '''
#     def __gt__(self, other):
#         '''  
#         Key for RRTX priority queue is ordered pair ( min(g(v), lmc(v)), g(v) )
#         (a, b) > (c, d) iff not a < c or (a == c and b < d)
#         '''
#         a = min(self.cost_to_goal, self.lmc)
#         c = min(other.cost_to_goal, other.lmc)
#         b = self.cost_to_goal
#         d = other.cost_to_goal
#         return not (a < c or (a == c and b < d))
    
#     def __eq__(self, other):
#         return min(self.cost_to_goal, self.lmc) == min(other.cost_to_goal, other.lmc)

# class _Wrapper:
#     def __init__(self, item, key):
#         self.item = item
#         self.key = key

#     def __lt__(self, other):
#         return self.key(self.item) < other.key(other.item)

#     def __eq__(self, other):
#         return self.key(self.item) == other.key(other.item)


# class KeyPriorityQueue(PriorityQueue):
#     def __init__(self, key):
#         self.key = key
#         super().__init__()

#     def _get(self):
#         wrapper = super()._get()
#         return wrapper.item

#     def _put(self, item):
#         super()._put(_Wrapper(item, self.key))

# def pq_comparator(node1: Node, node2: Node) -> int:
#     a = min(node1.cost_to_goal, node1.lmc)
#     c = min(node2.cost_to_goal, node2.lmc)
#     b = node1.cost_to_goal
#     d = node2.cost_to_goal
#     if a < c or (a == c and b < d):
#         return 1
#     elif a > c or (a == c and b > d):
#         return -1
#     else:
#         return 0
    

class RRTX:
    def __init__(self, x_start, x_goal, step_len, epsilon, 
                 goal_sample_rate, search_radius, iter_max):
        self.s_start = Node(x_start)
        self.s_goal = Node(x_goal)
        self.s_bot = self.s_start
        self.step_len = step_len
        self.epsilon = epsilon
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max
        self.vertices_coor = [[self.s_start.x, self.s_start.y]] # for faster nearest neighbor lookup
        self.tree_nodes = [self.s_start] # this is V_T in the paper
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
        self.gamma_FOS = 1.0 # factor of safety so that gamma > expression from Theorem 38 of RRT* paper
        self.update_gamma() # initialize gamma

    def planning(self):

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

            node_rand = self.random_node(self.goal_sample_rate)
            node_nearest = self.nearest(node_rand)
            node_new = self.saturate(node_nearest, node_rand) # this also sets cost_to_parent and cost_to_goal

            # REPLACE WITH EXTEND FUNCTION
            if node_new and not self.utils.is_collision(node_nearest, node_new):
                node_new.set_parent(new_parent=node_nearest)
                neighbor_indices = self.neighbor_indices(node_new)
                self.add_node(node_new)

                if neighbor_indices:
                    self.find_parent(node_new, neighbor_indices)
                    self.rewire(node_new, neighbor_indices)
            # END REPLACE
            
            # Add call to rewire here
            # Add call to reduce_inconsistency here

    def extend(self, node_new):
        neighbor_indices = self.neighbor_indices(node_new)
        self.find_parent(node_new, neighbor_indices)
        if self.parent == None:
            return 
        # child has already been added to parent's children
        for u in neighbor_indices:
            if not self.utils.is_collision(node_new, u):
                node_new.og_neighbor.add(u)
                u.in_neighbor.add(node_new)
            if not self.utils.is_collision(u, node_new):
                u.out_neighbor.add(node_new)
                node_new.og_neighbor.add(u)
                
    def update_obstacles(self, event):
        x, y = int(event.xdata), int(event.ydata)
        self.add_new_obstacle([x, y, 2])
        self.propagate_descendants()
        self.verify_queue(self.s_bot)

        # reduce inconsistency

    def add_new_obstacle(self, obs):
        print("Add circle obstacle at: s =", x, ",", "y =", y)
        self.obs_circle.append(obs)
        self.plotting.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle) # for plotting obstacles
        self.utils.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle) # for collision checking
        self.update_gamma() # free space volume changed, so gamma must change too

        # get all edges that intersect the new circle obstacle
        # DOES EDGE SET "E" INCLUDE EDGES THAT HAVE BEEN REMOVED PREVIOUSLY??? NOT ACCOUNTING RIGHT NOW
        E_O = [(v, u) for v in self.tree_nodes if (u:=v.parent) and utils.is_intersect_circle(*utils.get_ray(v, u), obs[:2], obs[2])]
        for v, u in E_O:
            v.infinite_dist_nodes.add(u)
            u.infinite_dist_nodes.add(v)
            if v.parent == u: # this check is currently irrelevant since u is always v's parent
                self.verify_orphan(v)
                # should theoretically check if the robot is on this edge now, but we do not

        heapq.heapify(self.Q) # reheapify after removing a bunch of elements and ruining queue

    def verify_orphan(self, v):
        # if v is in Q, remove it from Q and add it to orphan_nodes
        if v in self.Q:
            self.Q.remove(v)
            self.orphan_nodes.add(v)

    def propagate_descendants(self):
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
        # this does not do the updating, it is done after all changes are made (in propagate_descendants)
        # if v is in Q, update its cost and position, otherwise just add it
        try:
            self.Q.remove(v)
        except ValueError:
            pass
        heapq.heappush(self.Q, (v.get_key(), v))

    def reduce_inconsistency(self):

        while len(self.Q) > 0 and (heapq.nsmallest(1, self.Q)[0] < self.s_bot.get_key() \
                or self.s_bot.lmc != self.s_bot.cost_to_goal or np.isinf(self.s_bot.cost_to_goal) \
                or self.s_bot in self.Q):
            v = heapq.heappop(self.Q)[1]
        
        if v.cost_to_goal - v.lmc > self.epsilon:
            v.update_LMC()

            # TO BE CONTINUED

    def add_node(self, node_new):
        self.tree_nodes.append(node_new)
        self.vertices_coor.append([node_new.x, node_new.y])

    def saturate(self, node_start, node_goal):
        dist, theta = self.get_distance_and_angle(node_start, node_goal)
        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        return node_new

    def find_parent(self, node_new, neighbor_indices):
        costs = [self.get_new_cost(self.tree_nodes[i], node_new) for i in neighbor_indices]
        min_cost_index = int(np.argmin(costs))
        min_cost_neighbor_index = neighbor_indices[min_cost_index]
        if costs[min_cost_index] < node_new.cost_to_goal:
            node_new.set_parent(new_parent=self.tree_nodes[min_cost_neighbor_index])

    def rewire(self, node_new, neighbor_indices):
        for i in neighbor_indices:
            node_neighbor = self.tree_nodes[i]

            new_cost = self.get_new_cost(node_new, node_neighbor)
            if new_cost < node_neighbor.cost_to_goal:
                node_neighbor.set_parent(new_parent=node_new)

    def get_new_cost(self, node_start, node_end):
        dist = self.get_distance(node_start, node_end)
        return node_start.cost_to_goal + dist

    def random_node(self, goal_sample_rate):
        delta = self.utils.delta

        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.s_goal

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

        self.gamma = self.gamma_FOS * 2 * (1 + 1/self.d)**(1/self.d) * (mu_X_free/self.zeta_d)**(1/self.d) # optimality condition from Theorem 38 of RRT* paper

    def shrinking_ball_radius(self):
        '''
        Computes and returns the radius for the shrinking ball
        '''
        return min(self.step_len, self.gamma * (np.log(len(self.tree_nodes)) / len(self.tree_nodes))**(1/self.d))

    def neighbor_indices(self, node):
        
        nodes_coor = np.array(self.vertices_coor)
        node_coor = np.array([node.x, node.y]).reshape((1,2))
        dist_table = np.linalg.norm(nodes_coor - node_coor, axis=1)
        dist_table_index = [ind for ind in range(len(dist_table)) if dist_table[ind] <= self.search_radius and 
                            self.tree_nodes[ind] != node.parent and not self.utils.is_collision(node, self.tree_nodes[ind])]
        return dist_table_index

    def nearest(self, node):
        nodes_coor = np.array(self.vertices_coor)
        node_coor = np.array([node.x, node.y])
        dist_table = np.linalg.norm(nodes_coor - node_coor, axis=1)
        return self.tree_nodes[int(np.argmin(dist_table))]

    def extract_path(self, node_end):
        path = [[self.s_goal.x, self.s_goal.y]]
        node = node_end

        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path
    
    def cull_neighbors(self, v):
        for u in v.out_neighbors:
            if (self.get_distance(v,u) > self.search_radius) and (v.parent != u):
                v.out_neighbors.remove(u)
                u.in_neighbors.remove(v)
    
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

    rrtx = RRTX(x_start, x_goal, 5.0, 0.10, 20, 10000)
    rrtx.planning()


if __name__ == '__main__':
    main()
