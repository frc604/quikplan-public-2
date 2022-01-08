import matplotlib.pyplot as plt
import numpy as np

from field import Field
from robot import Robot
from quikplan import QuikPlan, StoppedPoseConstraint, StoppedXYConstraint, XYConstraint, XConstraint, YConstraint, GoalConstraint, SpeedConstraint
from helpers import write_to_csv

# Create the field
field = Field()

# Create the robot model
robot = Robot()

# Configure the optimizer
start_pose = (4.5, -1.7, 0)
qp = QuikPlan(
    field,
    robot,
    start_pose,
    [StoppedPoseConstraint(start_pose)],
)
waypoint2 = (3.8, -1.7, 0)
qp.add_waypoint(
    waypoint2,
    25,
    end_constraints=[StoppedXYConstraint(waypoint2)]
)

# Plan the trajectory
traj = qp.plan()
write_to_csv(traj, './output/03_ball.csv')

# Plot
field.plot_traj(robot, traj, './output/03_ball.png', save=True)

# Animate
field.anim_traj(robot, traj, '03_ball.gif', save_gif=False)
