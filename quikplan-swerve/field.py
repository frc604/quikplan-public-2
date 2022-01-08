import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from helpers import in2m

# (0, 0) at the center of the field
# +x points toward the opposing alliance station wall
# +y points left
class Field(object):
    LENGTH = 15.98  # m
    WIDTH = 8.21  # m

    OBSTACLES = [
        (2.55, -0.95, 0.25),  # Far post
        # (-2.55, 0.95, 0.25),  # Near post
        # (1.13, 2.47, 0.25),  # Left post
        # (-1.13, -2.47, 0.25),  # Right post
    ]  # (x, y, radius)

    BALLS = [
        (1.83, -3.40),  # Triple 1
        (0.91, -3.40),  # Triple 2
        (0.00, -3.40),  # Triple 3
        (0.96, -1.00),  # Middle 1
        (1.43, -0.80),  # Middle 2
        (1.90, -0.61),  # Middle 3
        (1.12, -0.06),  # Middle 4
        (1.59, 0.14),  # Middle 5
        (1.63, 3.16),  # Double 1
        (1.63, 3.63),  # Double 2
    ]
    BALL_RADIUS = in2m(3.5)

    GOAL = (LENGTH * 0.5, -1.6)

    def plot_field(self, ax):
        img = mpimg.imread('2021_field.png')
        ax.imshow(img, extent=[-self.LENGTH * 0.5, self.LENGTH * 0.5, -self.WIDTH * 0.5, self.WIDTH * 0.5])
        # Plot obstacles
        for x, y, r in self.OBSTACLES:
            ax.add_artist(plt.Circle((x, y), r, color='r'))
        # Plot balls
        for x, y in self.BALLS:
            ax.add_artist(plt.Circle((x, y), self.BALL_RADIUS, color='y'))

    def plot_traj(self, robot, traj, save_file, plot_mod=10, save=False):
        fig, ax = plt.subplots()
        
        # Plot field and path
        self.plot_field(ax)
        plt.scatter(traj[:, 1], traj[:, 2], marker='.', color=['g' if shoot else 'r' for shoot in traj[:, 7]])

        traj_size = traj.shape[0]
        for k in range(traj_size):
            if k % plot_mod == 0 or k == traj_size - 1:
                robot.plot(ax, traj[k, 1:4])

        fig.set_size_inches(18, 9)
        plt.savefig(save_file)
        plt.show()

    def anim_traj(self, robot, traj, save_file, save_gif=False):
        fig, ax = plt.subplots()

        # Plot field and path
        self.plot_field(ax)
        plt.scatter(traj[:, 1], traj[:, 2], marker='.', color=['g' if shoot else 'r' for shoot in traj[:, 7]])

        # Plot first pose
        robot.plot(ax, traj[0, 1:4])
        # Plot last pose
        robot.plot(ax, traj[-1, 1:4])

        # Animation function
        def animate(i):
            print("Rendering Frame: {}".format(i))
            # Hack to remove the old robot poses
            ax.collections = ax.collections[:7]
            robot.plot(ax, traj[i, 1:4])
            return ax.collections

        anim = animation.FuncAnimation(
            fig, animate, frames=traj.shape[0], interval=20, blit=True, repeat=True
        )
        if save_gif:
            anim.save("out.gif", writer="pillow", fps=50)
        plt.show()
