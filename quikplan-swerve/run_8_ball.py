import matplotlib.pyplot as plt
import numpy as np

from field import Field
from robot import Robot
from quikplan import QuikPlan, StoppedPoseConstraint, PoseConstraint, StoppedXYConstraint, XYConstraint, GoalConstraint, SpeedConstraint
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
    [(2.5, 4)],
)

waypoint2 = (3.8, -1.7  , 0)
qp.add_waypoint(
    waypoint2,
    15,
    end_constraints=[XYConstraint(waypoint2), GoalConstraint()]
)
waypoint2 = (3.77, -1.7, 0)
qp.add_waypoint(
    waypoint2,
    15,
    intermediate_constraints=[GoalConstraint(), SpeedConstraint(0.01)],
    end_constraints=[XYConstraint(waypoint2), GoalConstraint()]
)
waypoint4 = (1.5, -2.0, 0)
qp.add_waypoint(
    waypoint4,
    15,
    end_constraints=[XYConstraint(waypoint4)]
)

# Pickup first 3 balls
waypoint5 = (0.0, -1.3, -np.pi * 0.85)
qp.add_waypoint(
    waypoint5,
    15,
    end_constraints=[PoseConstraint(waypoint5)],
)
waypoint6 = (0.5, -1.0, -np.pi * 0.85)
qp.add_waypoint(
    waypoint6,
    15,
    end_constraints=[PoseConstraint(waypoint6)],
)
waypoint7 = (1.43, -0.60, -np.pi * 0.85)
qp.add_waypoint(
    waypoint7,
    15,
    end_constraints=[StoppedPoseConstraint(waypoint7)],
)
# Pickup next 2 balls
waypoint8 = (0.20, -0.60, -np.pi * 0.85)
qp.add_waypoint(
    waypoint8,
    15,
    end_constraints=[PoseConstraint(waypoint8)],
)
waypoint9 = (1.12, 0.1, -np.pi * 0.85)
qp.add_waypoint(
    waypoint9,
    15,
    end_constraints=[StoppedPoseConstraint(waypoint9)],
)
# Head to goal
waypoint10 = (1.7, -1.9, 0)
qp.add_waypoint(
    waypoint10,
    15,
    end_constraints=[StoppedXYConstraint((3.0, -1.9, 0)), GoalConstraint()],
)
# # Final Stop
# waypoint11 = (2.0, -1.9, 0)
# qp.add_waypoint(
#     waypoint11,
#     15,
#     intermediate_constraints=[GoalConstraint(), SpeedConstraint(0.1)],
#     end_constraints=[StoppedXYConstraint(waypoint11)],
# )

# Plan the trajectory
traj = qp.plan()
write_to_csv(traj, './output/08_ball.csv')

# Plot
field.plot_traj(robot, traj, './output/08_ball.png', save=True)

# Animate
field.anim_traj(robot, traj, '08_ball.gif', save_gif=False)
