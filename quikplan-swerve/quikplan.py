import casadi as ca
import numpy as np

from helpers import in2m, interp_state_vector


class BasePoseConstraint(object):
    def __init__(self, pose):
        self.pose = pose


class StoppedPoseConstraint(BasePoseConstraint):
    pass


class PoseConstraint(BasePoseConstraint):
    pass


class XConstraint(BasePoseConstraint):
    pass


class YConstraint(BasePoseConstraint):
    pass


class XYConstraint(BasePoseConstraint):
    pass


class StoppedXYConstraint(BasePoseConstraint):
    pass


class AngularConstraint(BasePoseConstraint):
    pass


class GoalConstraint(object):
    pass


# Hacky way to enforce stop time
class SpeedConstraint(object):
    def __init__(self, speed):
        self.speed = speed


class QuikPlan(object):
    BALL_VELOCITY = 30.0  # m/s
    SHOOTING_Y_TOL = 2.0  # m

    def __init__(self, field, robot, start_pose, start_constraints, shoot_times=[]):
        self._field = field
        self._robot = robot

        # Keep track of states as an initial guess
        self._states = np.zeros((1, 3))
        self._states[0, :] = start_pose

        # List of the number of control intervals between each waypoint. There are |sum(Ns) + 1| states.
        self._Ns = []

        # List of (state_idx, constraint) tuples
        self._constraints = []
        for c in start_constraints:
            self._constraints.append((0, c))

        # Time after which not to auto shoot
        self._shoot_times = shoot_times

    def add_waypoint(self, pose, N, intermediate_constraints=[], end_constraints=[]):
        # Linearly interpolate to initialize states
        # TODO: Fix excessive copying
        last_state = self._states[-1, :]
        new_states = np.zeros((N, 3))
        new_states[:, 0] = np.linspace(last_state[0], pose[0], N + 1)[1:]
        new_states[:, 1] = np.linspace(last_state[1], pose[1], N + 1)[1:]
        new_states[:, 2] = np.linspace(last_state[2], pose[2], N + 1)[1:]
        self._states = np.vstack((self._states, new_states))

        # Save constraints
        start_N = sum(self._Ns)
        for i in range(start_N, start_N + N + 1):
            for c in intermediate_constraints:
                self._constraints.append((i, c))
            if i == start_N + N:
                for c in end_constraints:
                    self._constraints.append((i, c))

        # Update Ns
        self._Ns.append(N)

    def plot_init(self, ax, mod=10):
        num_states = self._states.shape[0]
        for i in range(num_states):
            state = self._states[i]
            self._robot.plot(ax, state)

    def is_pointed_at_goal(self, interp_times, interp_x, interp_y, interp_theta, interp_xDot, interp_yDot):
        out = []
        for time, x, y, theta, xDot, yDot in zip(interp_times, interp_x, interp_y, interp_theta, interp_xDot, interp_yDot):
            ball_xDot = self.BALL_VELOCITY * np.cos(theta)
            ball_yDot = self.BALL_VELOCITY * np.sin(theta)
            ball_vel = np.array([ball_xDot, ball_yDot])
            robot_vel = np.array([xDot, yDot])
            vel_sum = robot_vel + ball_vel
            combined_theta = np.arctan2(vel_sum[1], vel_sum[0])

            goal_dx = self._field.GOAL[0] - x
            goal_dy = self._field.GOAL[1] - y
            goal_phi = np.arctan(goal_dy / goal_dx)

            aligned = np.abs(combined_theta - goal_phi) < np.deg2rad(1) and abs(goal_dy) < self.SHOOTING_Y_TOL
            in_time = False
            for interval_low, interval_high in self._shoot_times:
                if interval_low < time and time < interval_high:
                    in_time = True
            out.append(aligned and in_time)
        return out

    def plan(self):
        # Construct optimization problem
        opti = ca.Opti()
        N = sum(self._Ns)

        # State variables
        X = opti.variable(9, N + 1)
        xpos = X[0, :]
        ypos = X[1, :]
        theta = X[2, :]
        xDot = X[3, :]
        yDot = X[4, :]
        thetaDot = X[5, :]
        xDotDot = X[6, :]
        yDotDot = X[7, :]
        thetaDotDot = X[8, :]

        # Control variables
        U = opti.variable(3, N)
        xDotDotDot = U[0, :]
        yDotDotDot = U[1, :]
        thetaDotDotDot = U[2, :]

        # Total time variable per segment
        Ts = []
        dts = []
        for n in self._Ns:
            T = opti.variable()
            dt = T / n
            Ts.append(T)
            dts.append(dt)

            # Apply time constraint & initial guess
            opti.subject_to(T >= 0)
            opti.set_initial(T, 5)

        # Minimize time
        total_time = sum(Ts)
        opti.minimize(total_time)

        # Apply dynamic constriants
        start_n = 0
        for n, dt in zip(self._Ns, dts):
            end_n = start_n + n
            for k in range(start_n, end_n):
                x_next = X[:, k] + self._robot.dynamics_model(X[:, k], U[:, k]) * dt
                opti.subject_to(X[:, k + 1] == x_next)
            start_n = end_n

        # Apply state constraints
        for i, constraint in self._constraints:
            if type(constraint) in {StoppedPoseConstraint, PoseConstraint}:
                opti.subject_to(X[0, i] == constraint.pose[0])
                opti.subject_to(X[1, i] == constraint.pose[1])
            if type(constraint) in {StoppedPoseConstraint, PoseConstraint, AngularConstraint}:
                opti.subject_to(X[2, i] == constraint.pose[2])
            if type(constraint) in {StoppedPoseConstraint, StoppedXYConstraint}:
                opti.subject_to(X[3, i] == 0.0)
                opti.subject_to(X[4, i] == 0.0)
                opti.subject_to(X[5, i] == 0.0)
            if type(constraint) in {XConstraint, XYConstraint, StoppedXYConstraint}:
                opti.subject_to(X[0, i] == constraint.pose[0])
            if type(constraint) in {YConstraint, XYConstraint, StoppedXYConstraint}:
                opti.subject_to(X[1, i] == constraint.pose[1])
            if type(constraint) == GoalConstraint:
                # Ball velocity vector + robot velocity velocity vector should point at the goal.
                ball_xDot = self.BALL_VELOCITY * np.cos(theta[i])
                ball_yDot = self.BALL_VELOCITY * np.sin(theta[i])
                ball_vel = np.array([ball_xDot, ball_yDot])
                robot_vel = np.array([xDot[i], yDot[i]])
                vel_sum = robot_vel + ball_vel
                combined_theta = np.arctan2(vel_sum[1], vel_sum[0])

                goal_dx = self._field.GOAL[0] - xpos[i]
                goal_dy = self._field.GOAL[1] - ypos[i]
                goal_phi = np.arctan(goal_dy / goal_dx)
                opti.subject_to(combined_theta == goal_phi)
            if type(constraint) == SpeedConstraint:
                self._robot.apply_speed_constraint(opti, X, U, i, constraint.speed)

        # Apply module torque/friction constraints
        self._robot.apply_module_constraints(opti, X, U, N)

        # Apply obstacle constraints
        self._robot.apply_obstacle_constraints(opti, xpos, ypos, theta, self._field.OBSTACLES)
    
        # Set initial guess
        opti.set_initial(xpos, self._states[:, 0])
        opti.set_initial(ypos, self._states[:, 1])
        opti.set_initial(theta, self._states[:, 2])

        # Solve
        opti.solver("ipopt", {}, {"mu_init": 1e-3})
        sol = opti.solve()
        for t in Ts:
            print(sol.value(t))
        print(f"Total time: {sol.value(total_time)}")

        # Interpolate result
        times = [0.0]
        for n, t, dt in zip(self._Ns, Ts, dts):
            times += list(np.linspace(times[-1] + sol.value(dt), times[-1] + sol.value(t), n))

        interp_times, interp_x = interp_state_vector(times, sol.value(xpos), 0.02)
        _, interp_y = interp_state_vector(times, sol.value(ypos), 0.02)
        _, interp_theta = interp_state_vector(times, sol.value(theta), 0.02)
        _, interp_xDot = interp_state_vector(times, sol.value(xDot), 0.02)
        _, interp_yDot = interp_state_vector(times, sol.value(yDot), 0.02)
        _, interp_thetaDot = interp_state_vector(times, sol.value(thetaDot), 0.02)

        # Determine we are pointed at the goal at the interpolated states
        interp_pointed_at_goal = self.is_pointed_at_goal(interp_times, interp_x, interp_y, interp_theta, interp_xDot, interp_yDot)

        return np.transpose(np.vstack([interp_times, interp_x, interp_y, interp_theta, interp_xDot, interp_yDot, interp_thetaDot, interp_pointed_at_goal]))
