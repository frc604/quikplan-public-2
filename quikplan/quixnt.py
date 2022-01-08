from robot import Robot
import time as timeLib
from networktables import NetworkTables

from quikplan_live_search import plan

# logging.basicConfig(level=logging.DEBUG)

NetworkTables.initialize(server='10.6.4.2')
# NetworkTables.initialize(server='localhost')

while not NetworkTables.isConnected():
	print("Connecting.")
	timeLib.sleep(1)

print("Connected to {}!".format(NetworkTables.getRemoteAddress()))

timeLib.sleep(3)


nt = NetworkTables.getTable('quikplan')
targetTable = nt.getSubTable('obstacles')
trajectoryTable = nt.getSubTable('trajectory')

for key in trajectoryTable.getKeys():
	trajectoryTable.delete(key)
	timeLib.sleep(0.05)

targets = []

nt.putBoolean("Valid_Trajectory", False)


while len(targetTable.getKeys()) == 0:
	print("Waiting for targets.")
	timeLib.sleep(1)

for key in targetTable.getKeys():
	targets.append(targetTable.getNumberArray(key, (0, 0, 0)))


targets = sorted(targets, key=lambda target : target[0])

robot = Robot()

trajectory = plan(robot, targets, plot=True)

# Testing trajectory
# trajectory = [[], [], [], [], [], [], [], [], [], [], [], []]

# for i in range(0, 200):
# 	for x in range(0, 12):
# 		trajectory[x].append(i)

# for i in range(0, len(trajectory)):
# 	trajectory[i] = tuple(trajectory[i])

# trajectory = tuple(trajectory)

for (time, x, y, theta, v, w, vl, vr, al, ar, jl, jr) in zip(trajectory[0], trajectory[1], trajectory[2], trajectory[3], trajectory[4], trajectory[5], trajectory[6], trajectory[7], trajectory[8], trajectory[9], trajectory[10], trajectory[11]):
	trajectoryTable.putNumberArray(str(float(time)), list(map(float, [x, y, theta, v, w, vl, vr, al, ar, jl, jr])))
	timeLib.sleep(0.05)

nt.putBoolean("Valid_Trajectory", True)
timeLib.sleep(1)