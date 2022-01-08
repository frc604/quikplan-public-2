import matplotlib.pyplot as plt
import numpy as np

from field import Field
from robot import Robot
from quikplan import QuikPlan, StoppedPoseConstraint, PoseConstraint, StoppedXYConstraint, GoalConstraint, XYConstraint, SpeedConstraint
from helpers import write_to_csv

# Create the field
field = Field()

# Create the robot model
robot = Robot()

# Configure the optimizer
start_pose = (4.5, 3.2, 0)
qp = QuikPlan(
    field,
    robot,
    start_pose,
    [StoppedPoseConstraint(start_pose)],
    0.0,  # Shooter lockout
)
# Intake
waypoint0 = (2.4, 2.6, -np.pi * 0.25)
qp.add_waypoint(
    waypoint0,
    25,
    end_constraints=[StoppedPoseConstraint(waypoint0)]
)
waypoint1 = (2.0, 3.0, -np.pi * 0.25)
qp.add_waypoint(
    waypoint1,
    25,
    intermediate_constraints=[SpeedConstraint(1.0)],
    end_constraints=[PoseConstraint(waypoint1)]
)
waypoint2 = (2.0, 3.2, -np.pi * 0.35)
qp.add_waypoint(
    waypoint2,
    25,
    intermediate_constraints=[SpeedConstraint(1.0)],
    end_constraints=[StoppedPoseConstraint(waypoint2)]
)
# Go to shoot
waypoint1 = (3.5, 0.0, 0)
qp.add_waypoint(
    waypoint1,
    25,
    end_constraints=[StoppedXYConstraint(waypoint1), GoalConstraint()]
)
# waypoint2 = (4.0, -0.7, 0)
# qp.add_waypoint(
#     waypoint2,
#     25,
#     intermediate_constraints=[GoalConstraint(), SpeedConstraint(0.1)],
#     end_constraints=[StoppedXYConstraint(waypoint2)]
# )

# # Drive away
# waypoint1 = (5.0, 1.6, -np.pi)
# qp.add_waypoint(
#     waypoint1,
#     10,
#     end_constraints=[StoppedPoseConstraint(waypoint1)]
# )

# Plan the trajectory
traj = qp.plan()
write_to_csv(traj, './output/05_ball.csv')

# Plot
field.plot_traj(robot, traj, './output/05_ball.png', save=True)

# Animate
field.anim_traj(robot, traj, '05_ball.gif', save_gif=False)
