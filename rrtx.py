import os
import sys
import math
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
        self.cost_from_parent = 0.0
        self.cost_from_start = 0.0

class Edge:
    def __init__(self, n_p, n_c):
        self.parent = n_p
        self.child = n_c
        self.flag = "VALID"

class RRTX:
    def __init__(self, x_start, x_goal, step_len,
                 goal_sample_rate, search_radius, iter_max):
        self.s_start = Node(x_start)
        self.s_goal = Node(x_goal)
        self.s_bot = self.s_start
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max
        self.vertices = [self.s_start]
        self.vertices_coor = [[self.s_start.x, self.s_start.y]] # for faster nearest neighbour lookup
        self.edges = []
        self.path = []

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
                for node in self.vertices:
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
            node_new = self.saturate(node_nearest, node_rand) # this also sets cost_from_parent and cost_from_start

            if node_new and not self.utils.is_collision(node_nearest, node_new):
                self.set_parent_child(node_nearest, node_new)
                neighbour_indices = self.neighbour_indices(node_new)
                self.add_node(node_new)

                if neighbour_indices:
                    self.find_parent(node_new, neighbour_indices)
                    self.rewire(node_new, neighbour_indices)

    def update_obstacles(self, event):
        x, y = int(event.xdata), int(event.ydata)
        print("Add circle obstacle at: s =", x, ",", "y =", y)
        self.obs_add = [x, y, 2]
        self.obs_circle.append(self.obs_add)
        self.plotting.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle)
        self.utils.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle)
        self.update_gamma() # free space volume changed, so gamma must change too

    def add_node(self, node_new):
        self.vertices.append(node_new)
        self.vertices_coor.append([node_new.x, node_new.y])

    def saturate(self, node_start, node_goal):
        dist, theta = self.get_distance_and_angle(node_start, node_goal)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))

        return node_new

    def find_parent(self, node_new, neighbour_indices):
        costs = [self.get_new_cost(self.vertices[i], node_new) for i in neighbour_indices]
        min_cost_index = int(np.argmin(costs))
        min_cost_neighbour_index = neighbour_indices[min_cost_index]
        if costs[min_cost_index] < node_new.cost_from_start:
            self.change_parent(self.vertices[min_cost_neighbour_index], node_new)

    def rewire(self, node_new, neighbour_indices):
        for i in neighbour_indices:
            node_neighbour = self.vertices[i]

            new_cost = self.get_new_cost(node_new, node_neighbour)
            if new_cost < node_neighbour.cost_from_start:
                self.change_parent(node_new, node_neighbour)

    def set_parent_child(self, parent, child):
        child.parent = parent
        parent.children.add(child)
        dist = self.get_distance(child, parent)
        child.cost_from_parent = dist
        child.cost_from_start = parent.cost_from_start + dist

    def change_parent(self, new_parent, child):
        old_parent = child.parent
        old_parent.children.remove(child)
        child.parent = new_parent
        child.cost_from_parent = self.get_distance(child, new_parent)
        new_parent.children.add(child)
        self.update_costs_recursive(child)

    def update_costs_recursive(self, node):
        if node.parent:
            node.cost_from_start = node.parent.cost_from_start + node.cost_from_parent
            for child in node.children:
                self.update_costs_recursive(child)

    def get_new_cost(self, node_start, node_end):
        dist = self.get_distance(node_start, node_end)
        return node_start.cost_from_start + dist

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
        return min(self.step_len, self.gamma * (np.log(len(self.vertices)) / len(self.vertices))**(1/self.d))

    def neighbour_indices(self, node):
        nodes_coor = np.array(self.vertices_coor)
        node_coor = np.array([node.x, node.y]).reshape((1,2))
        dist_table = np.linalg.norm(nodes_coor - node_coor, axis=1)
        dist_table_index = [ind for ind in range(len(dist_table)) if dist_table[ind] <= self.search_radius and 
                            self.vertices[ind] != node.parent and not self.utils.is_collision(node, self.vertices[ind])]
        return dist_table_index

    def nearest(self, node):
        nodes_coor = np.array(self.vertices_coor)
        node_coor = np.array([node.x, node.y])
        dist_table = np.linalg.norm(nodes_coor - node_coor, axis=1)
        return self.vertices[int(np.argmin(dist_table))]

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

    rrtx = RRTX(x_start, x_goal, 5.0, 0.10, 20, 10000)
    rrtx.planning()


if __name__ == '__main__':
    main()
