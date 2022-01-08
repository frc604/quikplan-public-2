import matplotlib.pyplot as plt
import numpy as np

from field import Field
from robot import Robot
from quikplan import QuikPlan, StoppedPoseConstraint, PoseConstraint, StoppedXYConstraint, XYConstraint, XConstraint, YConstraint, GoalConstraint, SpeedConstraint
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
    [StoppedPoseConstraint(start_pose)]
)
# Pickup first 3 balls
waypoint2 = (2.4, -3.3, 0)
qp.add_waypoint(
    waypoint2,
    15,
    end_constraints=[XYConstraint(waypoint2), GoalConstraint()]
)
waypoint3 = (0.2, -3.3, 0)
qp.add_waypoint(
    waypoint3,
    15,
    intermediate_constraints=[YConstraint(waypoint3), GoalConstraint(), SpeedConstraint(1.5)],
    end_constraints=[XConstraint(waypoint3)]
)
# Finish shooting first 6 balls
waypoint4 = (0.5, -2.5, 0)
qp.add_waypoint(
    waypoint4,
    15,
    intermediate_constraints=[GoalConstraint(), SpeedConstraint(1.5)],
    end_constraints=[XYConstraint(waypoint4)],
)
# Pickup next 3 balls
waypoint5 = (0.0, -1.5, -np.pi * 0.85)
qp.add_waypoint(
    waypoint5,
    15,
    end_constraints=[PoseConstraint(waypoint5)],
)
waypoint6 = (0.5, -1.20, -np.pi * 0.85)
qp.add_waypoint(
    waypoint6,
    15,
    end_constraints=[PoseConstraint(waypoint6), SpeedConstraint(1.0)],
)
waypoint7 = (1.43, -0.80, -np.pi * 0.85)
qp.add_waypoint(
    waypoint7,
    15,
    intermediate_constraints=[SpeedConstraint(1.0)],
    end_constraints=[StoppedPoseConstraint(waypoint7)],
)
# Pickup next 2 balls
waypoint8 = (0.20, -0.80, -np.pi * 0.85)
qp.add_waypoint(
    waypoint8,
    15,
    end_constraints=[PoseConstraint(waypoint8)],
)
waypoint9 = (1.12, -0.06, -np.pi * 0.85)
qp.add_waypoint(
    waypoint9,
    15,
    intermediate_constraints=[SpeedConstraint(1.0)],
    end_constraints=[StoppedPoseConstraint(waypoint9)],
)
# Head to goal
waypoint10 = (1.7, -1.9, 0)
qp.add_waypoint(
    waypoint10,
    15,
    end_constraints=[XYConstraint(waypoint10), GoalConstraint()],
)
# Final Stop
waypoint11 = (2.0, -1.9, 0)
qp.add_waypoint(
    waypoint11,
    15,
    intermediate_constraints=[GoalConstraint(), SpeedConstraint(0.1)],
    end_constraints=[StoppedXYConstraint(waypoint11)],
)

# Plan the trajectory
traj = qp.plan()
write_to_csv(traj, './output/11_ball.csv')

# Plot
field.plot_traj(robot, traj, './output/11_ball.png', save=True)

# Animate
field.anim_traj(robot, traj, '11_ball.gif', save_gif=False)
