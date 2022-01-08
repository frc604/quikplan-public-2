import matplotlib.pyplot as plt
import numpy as np

from field import Field
from robot import Robot
from quikplan import QuikPlan, StoppedPoseConstraint, PoseConstraint, StoppedXYConstraint, XYConstraint, XConstraint, YConstraint, GoalConstraint
from helpers import write_to_csv

# Create the field
field = Field()

# Create the robot model
robot = Robot()

# Configure the optimizer
start_pose = (4.5, 3.4, 0)
qp = QuikPlan(
    field,
    robot,
    start_pose,
    [StoppedPoseConstraint(start_pose)]
)
waypoint0 = (2.1, 3.4, 0)
qp.add_waypoint(
    waypoint0,
    10,
    end_constraints=[StoppedPoseConstraint(waypoint0)]
)
waypoint1 = (3.5, -0.1, 0)
qp.add_waypoint(
    waypoint1,
    10,
    end_constraints=[XYConstraint(waypoint1), GoalConstraint()]
)
waypoint2 = (2.7, -3.3, 0)
qp.add_waypoint(
    waypoint2,
    10,
    intermediate_constraints=[GoalConstraint()],
    end_constraints=[XYConstraint(waypoint2), GoalConstraint()]
)
waypoint3 = (0.2, -3.3, 0)
qp.add_waypoint(
    waypoint3,
    10,
    intermediate_constraints=[YConstraint(waypoint3), GoalConstraint()],
    end_constraints=[XConstraint(waypoint3)]
)
waypoint4 = (1.5, -2.5, 0)
qp.add_waypoint(
    waypoint4,
    10,
    intermediate_constraints=[GoalConstraint()],
    end_constraints=[XYConstraint(waypoint4)],
)
waypoint5 = (0.5, -2.0, -np.pi * 0.5)
qp.add_waypoint(
    waypoint5,
    10,
    end_constraints=[PoseConstraint(waypoint5)],
)
waypoint6 = (1.4, -0.9, -np.pi * 0.85)
qp.add_waypoint(
    waypoint6,
    10,
    end_constraints=[StoppedPoseConstraint(waypoint6)],
)
waypoint7 = (0.4, -0.8, -np.pi * 0.85)
qp.add_waypoint(
    waypoint7,
    10,
    end_constraints=[PoseConstraint(waypoint7)],
)
waypoint8 = (1.3, -0.2, -np.pi * 0.85)
qp.add_waypoint(
    waypoint8,
    10,
    end_constraints=[PoseConstraint(waypoint8)],
)
waypoint9 = (1.7, -1.9, 0)
qp.add_waypoint(
    waypoint9,
    10,
    end_constraints=[StoppedXYConstraint(waypoint9), GoalConstraint()],
)

# Plan the trajectory
traj = qp.plan()
write_to_csv(traj, './output/13_ball.csv')

# Plot
field.plot_traj(robot, traj, './output/13_ball.png', save=True)

# Animate
field.anim_traj(robot, traj, '13_ball.gif', save_gif=False)
