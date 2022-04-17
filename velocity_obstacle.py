import numpy as np
import matplotlib.pyplot as plt

import env, plotting

class Velocity_Obstacle:
    def __init__(self, start, goal, robot_radius, timestep, iter_max):
        self.start = np.append(np.array(start), [0, 0]) # x, y, velx, vely
        self.goal = np.append(np.array(goal), [0, 0])
        self.robot_radius = robot_radius
        self.timestep = timestep
        self.iter_max = iter_max
        self.robot_state = self.start
        self.vmax = 0.2
        self.desired_vel = 0
        self.obstacle_robots = [[]]
        
        self.obstacle_FOS = 0.5
        self.obstacle_radius = 10
        
        self.env = env.Env()
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary
        
        self.plotting = plotting.Plotting(start, goal)
    
    def set_other_robots(self, other_robots):
        self.other_robots = other_robots
        self.obstacle_robots = []
        for robot in other_robots:
            self.obstacle_robots.append(robot.robot_state)
            
    
    def update_other_robots(self):
        for idx,robot in self.other_robots:
            self.obstacle_robots[idx] = robot.robot_state
    
    def step(self):
        # if reached goal, stop and return
        if self.robot_state[0] - self.goal[0] < 0.01 and self.robot_state[1] - self.goal[1] < 0.01:
            return
        
        self.update_other_robots()
        # MAIN ALGORITHM
        self.compute_desired_velocity() # find straight line to goal velocity
        self.compute_velocity() # find velocity that satisfies obstacle constraints and is closest to desired velocity
        self.update_state() # update robot state
        
    
    def single_robot_simulation(self):
        # start with plotting stuff
                    
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.suptitle('Velocity Obstacle Simulation')
        self.ax.set_xlim(self.env.x_range[0], self.env.x_range[1]+1)
        self.ax.set_ylim(self.env.y_range[0], self.env.y_range[1]+1)
        self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show(block=False)
        plt.pause(0.1)
        self.fig.canvas.blit(self.ax.bbox)
        
        
        for _ in range(self.iter_max):
            self.fig.canvas.restore_region(self.bg)
            self.plotting.plot_env(self.ax)
            self.plotting.plot_robot(self.ax, self.robot_state[:2], self.robot_radius)

            self.fig.canvas.blit(self.ax.bbox)
            #plt.pause(0.1)
            
            
            self.compute_desired_velocity() # find straight line to goal velocity
            self.compute_velocity() # find velocity that satisfies obstacle constraints and is closest to desired velocity
            self.update_state() # update robot state
            

    
    
    def compute_desired_velocity(self):
        disp_vec = (self.goal - self.robot_state)[:2]
        norm = np.linalg.norm(disp_vec)
        if norm < self.robot_radius / 5:
            return np.zeros(2)
        disp_vec = disp_vec / norm
        np.shape(disp_vec)
        
        self.desired_vel = self.vmax * disp_vec
    
    def compute_velocity(self):
        pA = self.robot_state[:2]
        vA = self.robot_state[-2:]
        # Compute the constraints
        # for each velocity obstacles
        robot_obstacles, circle_obstacles = self.nearby_obstacles()
        number_of_obstacles = len(robot_obstacles) + len(circle_obstacles)
        Amat = np.empty((number_of_obstacles * 2, 2))
        bvec = np.empty((number_of_obstacles * 2))
        
        for i in range(len(robot_obstacles)):
            # other robot obstacles
            obstacle = robot_obstacles[:, i] # obstacle = [x , y, vx, vy]
            pB = obstacle[:2]
            vB = obstacle[2:]
            dispBA = pA - pB 
            distBA = np.linalg.norm(dispBA) # displacement vector between robot and obstacle
            thetaBA = np.arctan2(dispBA[1], dispBA[0])
            if 2*(self.robot_radius + self.obstacle_FOS) * self.robot_radius > distBA: # if robot is close enough to obstacle
                distBA = 2*(self.robot_radius + self.obstacle_FOS)*self.robot_radius
            phi_obst = np.arcsin(2.2*self.robot_radius/distBA)
            phi_left = thetaBA + phi_obst
            phi_right = thetaBA - phi_obst

            # VO
            translation = vB
            Atemp, btemp = self.create_constraints(translation, phi_left, "left")
            Amat[i*2, :] = Atemp
            bvec[i*2] = btemp
            Atemp, btemp = self.create_constraints(translation, phi_right, "right")
            Amat[i*2 + 1, :] = Atemp
            bvec[i*2 + 1] = btemp
        
        for j in range(len(circle_obstacles)):
            # circle obstacle
            obstacle = circle_obstacles[j]
            pB = np.array(obstacle[:2])
            vB = np.array([0,0]) # static obstacle
            dispBA = pA - pB # distance between robot and closest edge of obstacle
            distBA = np.linalg.norm(dispBA) - obstacle[2] # displacement vector between robot and obstacle
            thetaBA = np.arctan2(dispBA[1], dispBA[0])
            if (2*self.robot_radius + self.obstacle_FOS) * self.robot_radius > distBA: # if robot is close enough to obstacle
                distBA = (2*self.robot_radius + self.obstacle_FOS)*self.robot_radius
            phi_obst = np.arcsin((2*self.robot_radius + self.obstacle_FOS)*self.robot_radius/distBA)
            phi_left = thetaBA + phi_obst
            phi_right = thetaBA - phi_obst

            # VO
            i = j + np.shape(self.obstacle_robots)[1]
            translation = vB
            Atemp, btemp = self.create_constraints(translation, phi_left, "left")
            Amat[i*2, :] = Atemp
            bvec[i*2] = btemp
            Atemp, btemp = self.create_constraints(translation, phi_right, "right")
            Amat[i*2 + 1, :] = Atemp
            bvec[i*2 + 1] = btemp

        # Create search-space
        th = np.linspace(0, 2*np.pi, 20)
        vel = np.linspace(0, self.vmax, 5)

        vv, thth = np.meshgrid(vel, th)

        vx_sample = (vv * np.cos(thth)).flatten()
        vy_sample = (vv * np.sin(thth)).flatten()

        v_sample = np.stack((vx_sample, vy_sample))

        v_satisfying_constraints = self.check_constraints(v_sample, Amat, bvec)

        # Objective function
        size = np.shape(v_satisfying_constraints)[1]
        diffs = v_satisfying_constraints - \
            ((self.desired_vel).reshape(2, 1) @ np.ones(size).reshape(1, size))
        norm = np.linalg.norm(diffs, axis=0)
        min_index = np.where(norm == np.amin(norm))[0][0]
        self.cmd_vel = (v_satisfying_constraints[:, min_index])
        
    def check_constraints(self, v_sample, Amat, bvec):
        length = np.shape(bvec)[0]

        for i in range(int(length/2)):
            v_sample = self.check_inside(v_sample, Amat[2*i:2*i+2, :], bvec[2*i:2*i+2])

        return v_sample
    
    def check_inside(self, v, Amat, bvec):
        v_out = []
        for i in range(np.shape(v)[1]):
            if not ((Amat @ v[:, i] < bvec).all()):
                v_out.append(v[:, i])
        return np.array(v_out).T
    
    def create_constraints(self, translation, angle, side):
        origin = np.array([0, 0, 1])
        point = np.array([np.cos(angle), np.sin(angle)])
        line = np.cross(origin, point)
        line = self.translate_line(line, translation)
        if side == "left":
            line *= -1

        A = line[:2]
        b = -line[2]

        return A, b
    
    def translate_line(self, line, translation):
        matrix = np.eye(3)
        matrix[2, :2] = -translation[:2]
        return matrix @ line
    
    def update_state(self):
        new_state = np.empty((4))
        new_state[:2] = self.robot_state[:2] + self.cmd_vel * self.timestep
        new_state[-2:] = self.cmd_vel
        self.robot_state = new_state
    
    def nearby_obstacles(self):
        # Finds obstacles within a certain radius of the robot
        # Returns a list of obstacles
        robot_obstacles = []
        circle_obstacles = []
        for i in range(np.shape(self.obstacle_robots)[1]):
            if np.linalg.norm(self.obstacle_robots[:2,i] - self.robot_state[:2]) < self.obstacle_radius:
                robot_obstacles.append(self.obstacle_robots[:,i])
        for i in range(np.shape(self.obs_circle)[0]):
            if np.linalg.norm(self.obs_circle[i][:2] - self.robot_state[:2]) < self.obstacle_radius:
                circle_obstacles.append(self.obs_circle[i])
        return robot_obstacles, circle_obstacles
        
if __name__ == "__main__":
    start = (36,30)
    goal = (37,18)
    vel_obs = Velocity_Obstacle(start, goal, robot_radius = 0.5, timestep = 1, iter_max=10000)
    vel_obs.single_robot_simulation()