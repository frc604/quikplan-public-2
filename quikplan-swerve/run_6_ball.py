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
start_pose = (4.5, -3.4, 0)
qp = QuikPlan(
    field,
    robot,
    start_pose,
    [StoppedPoseConstraint(start_pose)],
    [(3, 6)]
)
waypoint2 = (3.2, -3.3, 0)
qp.add_waypoint(
    waypoint2,
    25,
    end_constraints=[XYConstraint(waypoint2), GoalConstraint()]
)
waypoint2 = (3.15, -3.3, 0)
qp.add_waypoint(
    waypoint2,
    5,
    intermediate_constraints=[GoalConstraint(), SpeedConstraint(0.01)],
    end_constraints=[StoppedXYConstraint(waypoint2), GoalConstraint()]
)
waypoint3 = (0.2, -3.3, 0)
qp.add_waypoint(
    waypoint3,
    25,
    intermediate_constraints=[YConstraint(waypoint3), GoalConstraint(), SpeedConstraint(1.5)],
    end_constraints=[XConstraint(waypoint3)]
)
waypoint4 = (2.0, -3.3, 0)
qp.add_waypoint(
    waypoint4,
    25,
    intermediate_constraints=[GoalConstraint()],
    end_constraints=[StoppedXYConstraint(waypoint4)],
)

# Plan the trajectory
traj = qp.plan()
write_to_csv(traj, './output/06_ball.csv')

# Plot
field.plot_traj(robot, traj, './output/06_ball.png', save=True)

# Animate
field.anim_traj(robot, traj, '06_ball.gif', save_gif=False)
