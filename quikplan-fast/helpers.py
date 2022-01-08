from dataclasses import dataclass
from functools import lru_cache
import os
from robot import (
    DEFAULT_MU,
    DEFAULT_MAX_RPM,
    DEFAULT_MAX_TORQUE,
    DEFAULT_OBSTACLE_BUFFER,
)
import matplotlib as mpl
import matplotlib.animation as animation
import numpy as np
import csv
import re
import pylab as plt
from PyQt5 import QtCore, QtWidgets


PATH_OBSTACLES = {
    "galactic-search-A-red": ["C3", "D5", "A6"],
    "galactic-search-A-blue": ["E6", "B7", "C9"],
    "galactic-search-B-red": ["B3", "D5", "B7"],
    "galactic-search-B-blue": ["D6", "B8", "D10"],
    "barrel-racing": ["B1", "B2", "B8", "D1", "D2", "D5", "D10"],
    "slalom": ["B1", "B2", "D1", "D2", "D4", "D5", "D6", "D7", "D8", "D10"],
    "slalom-simplified": ["B2", "D2", "D4", "D8", "D10"],
    "bounce": [
        "B1",
        "B2",
        "B4",
        "B5",
        "B7",
        "B8",
        "B10",
        "B11",
        "D1",
        "D2",
        "D3",
        "D5",
        "D7",
        "D8",
        "D10",
        "D11",
        "E3",
    ],
    "bounce-simplified": ["B2", "B10", "D5", "D7", "D8"],
    "hyperdrive": [
        "A6",
        "B1",
        "B3",
        "B4",
        "B6",
        "B7",
        "B9",
        "B11",
        "C9",
        "D1",
        "D3",
        "D4",
        "D6",
        "D7",
        "D8",
        "D9",
        "D10",
    ],
}


@dataclass(eq=True, frozen=True)
class OptiParams:
    mu: float = DEFAULT_MU
    max_torque: float = DEFAULT_MAX_TORQUE
    max_rpm: float = DEFAULT_MAX_RPM
    obstacle_buffer: float = DEFAULT_OBSTACLE_BUFFER


@dataclass(eq=True, frozen=True)
class Trajectory:
    type: str
    name: str
    params: OptiParams
    time: float = 0.0


def in2m(inches):
    # Inches to meters
    return inches * 2.54 / 100


def rotate_around_origin(point, theta):
    x, y = point
    return (
        x * np.cos(theta) - y * np.sin(theta),
        y * np.cos(theta) + x * np.sin(theta),
    )


def transform_geometry(geometry, pose):
    x, y, theta = pose
    transformed_geometry = []
    for point1, point2 in geometry:
        new_point1 = rotate_around_origin(point1, theta) + np.array([x, y])
        new_point2 = rotate_around_origin(point2, theta) + np.array([x, y])
        transformed_geometry.append((new_point1, new_point2))
    return transformed_geometry


def plot_robot(ax, pose, robot_geometry, robot_axis_plot_size):
    # Plot robot geometry
    ax.add_collection(
        mpl.collections.LineCollection(
            transform_geometry(robot_geometry, pose), color="k"
        )
    )
    # Plot robot axes
    ax.add_collection(
        mpl.collections.LineCollection(
            transform_geometry([[(0, 0), (robot_axis_plot_size, 0)]], pose),
            color="r",
        )
    )
    ax.add_collection(
        mpl.collections.LineCollection(
            transform_geometry([[(0, 0), (0, robot_axis_plot_size)]], pose),
            color="g",
        )
    )


def anim_traj(
    title,
    xs,
    ys,
    thetas,
    obstacles,
    robot_geometry,
    robot_axis_plot_size,
    timestep,
    targets=[],
    goal=None,
    limits=(-1, in2m(360) + 1, -1, in2m(180) + 1),
    draw_bounds=True,
):
    fig, ax = plt.subplots()
    num_states = len(xs)
    if draw_bounds:
        if draw_bounds:
            ax.add_patch(
                mpl.patches.Rectangle(
                    (0, 0),
                    in2m(360),
                    in2m(180),
                    linewidth=1,
                    edgecolor="k",
                    facecolor="none",
                )
            )
    for obx, oby, obr in obstacles:
        ax.add_artist(plt.Circle((obx, oby), obr, color="r"))
    for tx, ty, tr in targets:
        ax.add_artist(plt.Circle((tx, ty), tr, color="g"))
    if goal is not None:
        ax.add_collection(
            mpl.collections.LineCollection(
                [goal],
                color="r",
                linewidths=4,
            )
        )
    plt.scatter(xs, ys, marker=".")

    # Plot first pose
    plot_robot(ax, (xs[0], ys[0], thetas[0]), robot_geometry, robot_axis_plot_size)

    # Plot last pose
    plot_robot(
        ax,
        (xs[num_states - 1], ys[num_states - 1], thetas[num_states - 1]),
        robot_geometry,
        robot_axis_plot_size,
    )

    # Animation function
    def animate(i):
        pose = list(zip(xs, ys, thetas))[i]
        # Hack to remove the old robot poses
        ax.collections = ax.collections[:7]

        plot_robot(ax, pose, robot_geometry, robot_axis_plot_size)
        return ax.collections

    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.xlim(limits[0], limits[1])
    plt.ylim(limits[2], limits[3])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)

    return animation.FuncAnimation(
        fig, animate, frames=num_states, interval=timestep, blit=True, repeat=True
    )


def interp_state_vector(times, states, new_dt):
    interp_times = np.arange(0, times[-1], new_dt)
    return np.interp(interp_times, times, states[:-1])


def plot_traj(
    title,
    xs,
    ys,
    thetas,
    obstacles,
    robot_geometry,
    robot_axis_plot_size,
    targets=[],
    goal=None,
    limits=(-1, in2m(360) + 1, -1, in2m(180) + 1),
    draw_bounds=True,
    invert=False,
    robot_plot_mod=5,
):
    fig, ax = plt.subplots()
    if draw_bounds:
        ax.add_patch(
            mpl.patches.Rectangle(
                (0, 0),
                in2m(360),
                in2m(180),
                linewidth=1,
                edgecolor="k",
                facecolor="none",
            )
        )
    for obx, oby, obr in obstacles:
        ax.add_artist(plt.Circle((obx, oby), obr, color="r"))
    for tx, ty, tr in targets:
        ax.add_artist(plt.Circle((tx, ty), tr, color="g"))
    if goal is not None:
        ax.add_collection(
            mpl.collections.LineCollection(
                [goal],
                color="r",
                linewidths=4,
            )
        )
    plt.scatter(xs, ys, marker=".")
    for i, pose in enumerate(zip(xs, ys, thetas)):
        if robot_plot_mod is None:
            if i == 0:
                plot_robot(ax, pose, robot_geometry, robot_axis_plot_size)
        else:
            if i % robot_plot_mod == 0:
                plot_robot(ax, pose, robot_geometry, robot_axis_plot_size)
        # Always plot last pose
        if i == len(xs) - 1:
            plot_robot(ax, pose, robot_geometry, robot_axis_plot_size)
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.xlim(limits[0], limits[1])
    plt.ylim(limits[2], limits[3])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)

    if invert:
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()


def ids2obstales(ids, diameter):
    obstacles = []
    for id in ids:
        id = id.upper()
        row = ord(id[0]) - 64
        col = int(id[1:])
        x = in2m(30) * col
        y = in2m(180) - row * in2m(30)
        obstacles.append((x, y, diameter / 2))
    return obstacles


def create_obstacles(pathname):
    OBSTACLE_DIA_IN = 3  # inches
    return ids2obstales(PATH_OBSTACLES[pathname], in2m(OBSTACLE_DIA_IN))


def write_to_csv(filename, robot, times, xs, ys, thetas, vls, vrs, als, ars, jls, jrs):
    output = []
    for (time, x, y, theta, v, w, vl, vr, al, ar, jl, jr) in zip(
        times,
        xs,
        ys,
        thetas,
        ((vls + vrs) / 2),
        ((vrs - vls) / robot.TRACK_WIDTH),
        vls,
        vrs,
        als,
        ars,
        jls,
        jrs,
    ):
        output.append([time, x, y, theta, v, w, vl, vr, al, ar, jl, jr])

    with open("./{}.csv".format(filename), "w") as outfile:
        writer = csv.writer(outfile, delimiter=",")
        writer.writerows(output)
    outfile.close()


def read_trajectory(trajectory):
    # x, y, theta, v, w, vl, vr, al, ar, jl, jr = ([] for i in range(11))
    xs, ys, thetas, vs, ws, vls, vrs, als, ars = ([] for i in range(9))
    with open(
        "./trajectories/{}/{}".format(trajectory.type, trajectory.name)
    ) as inputfile:
        reader = csv.reader(inputfile, delimiter=",")
        for row in reader:
            xs.append(float(row[1]))
            ys.append(float(row[2]))
            thetas.append(float(row[3]))
            vs.append(float(row[4]))
            ws.append(float(row[5]))
            vls.append(float(row[6]))
            vrs.append(float(row[7]))
            als.append(float(row[8]))
            ars.append(float(row[9]))
            # jl.append(float(row[10]))
            # jr.append(float(row[11]))
    return (xs, ys, thetas, vls, vrs, als, ars)


@lru_cache(maxsize=128)
def index_trajectories():
    trajectories = {}
    for root, dirs, files in os.walk("trajectories"):
        for file in files:
            params = file.split("-")
            if not params[0] in trajectories:
                trajectories[params[0]] = []
            trajectories[params[0]].append(
                Trajectory(
                    params[0],
                    file,
                    OptiParams(
                        float(re.findall(r"[-+]?\d*\.\d+|\d+", params[2])[0]),
                        float(re.findall(r"[-+]?\d*\.\d+|\d+", params[3])[0]),
                        float(re.findall(r"[-+]?\d*\.\d+|\d+", params[4])[0]),
                        float(re.findall(r"[-+]?\d*\.\d+|\d+", params[1])[0]),
                    ),
                    float(re.findall(r"[-+]?\d*\.\d+|\d+", params[5])[0]),
                )
            )
    return trajectories


class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())

        self.dragging = None
        self.connect()

    def connect(self):
        self.cidpress = self.line.figure.canvas.mpl_connect(
            "button_press_event", self.on_press
        )
        self.cidrelease = self.line.figure.canvas.mpl_connect(
            "button_release_event", self.on_release
        )
        self.cidmotion = self.line.figure.canvas.mpl_connect(
            "motion_notify_event", self.on_motion
        )

    def disconnect(self):
        self.line.figure.canvas.mpl_disconnect(self.cidpress)
        self.line.figure.canvas.mpl_disconnect(self.cidrelease)
        self.line.figure.canvas.mpl_disconnect(self.cidmotion)

    def on_press(self, event):
        if event.inaxes != self.line.axes:
            return

        point_threshold = 0.1

        # Find nearest point to cursor
        min_distance = float("inf")
        for i, point in enumerate(zip(self.xs, self.ys)):
            x, y = point
            distance = np.sqrt((event.xdata - x) ** 2 + (event.ydata - y) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_index = i

        if event.button == 1:
            if min_distance < point_threshold:
                self.dragging = closest_index
            else:
                self.xs.append(event.xdata)
                self.ys.append(event.ydata)
                self.line.set_data(self.xs, self.ys)
                print(self.line.get_data())
                self.line.figure.canvas.draw()
        elif event.button == 3:
            if min_distance < point_threshold:
                self.xs.pop(closest_index)
                self.ys.pop(closest_index)
                self.line.set_data(self.xs, self.ys)
                self.line.figure.canvas.draw()

    def on_motion(self, event):
        if self.dragging is None:
            return
        if event.inaxes != self.line.axes:
            return

        self.xs[self.dragging] = event.xdata
        self.ys[self.dragging] = event.ydata
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

    def on_release(self, event):
        self.dragging = None
        self.line.figure.canvas.draw()


class QToaster(QtWidgets.QFrame):
    closed = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(QToaster, self).__init__(*args, **kwargs)
        QtWidgets.QHBoxLayout(self)

        self.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)

        self.setStyleSheet(
            """
            QToaster {
                border: 1px solid black;
                border-radius: 4px; 
                background: palette(window);
            }
        """
        )
        # alternatively:
        # self.setAutoFillBackground(True)
        # self.setFrameShape(self.Box)

        self.timer = QtCore.QTimer(singleShot=True, timeout=self.hide)

        if self.parent():
            self.opacityEffect = QtWidgets.QGraphicsOpacityEffect(opacity=0)
            self.setGraphicsEffect(self.opacityEffect)
            self.opacityAni = QtCore.QPropertyAnimation(self.opacityEffect, b"opacity")
            # we have a parent, install an eventFilter so that when it's resized
            # the notification will be correctly moved to the right corner
            self.parent().installEventFilter(self)
        else:
            # there's no parent, use the window opacity property, assuming that
            # the window manager supports it; if it doesn't, this won'd do
            # anything (besides making the hiding a bit longer by half a second)
            self.opacityAni = QtCore.QPropertyAnimation(self, b"windowOpacity")
        self.opacityAni.setStartValue(0.0)
        self.opacityAni.setEndValue(1.0)
        self.opacityAni.setDuration(100)
        self.opacityAni.finished.connect(self.checkClosed)

        self.corner = QtCore.Qt.TopLeftCorner
        self.margin = 10

    def checkClosed(self):
        # if we have been fading out, we're closing the notification
        if self.opacityAni.direction() == self.opacityAni.Backward:
            self.close()

    def restore(self):
        # this is a "helper function", that can be called from mouseEnterEvent
        # and when the parent widget is resized. We will not close the
        # notification if the mouse is in or the parent is resized
        self.timer.stop()
        # also, stop the animation if it's fading out...
        self.opacityAni.stop()
        # ...and restore the opacity
        if self.parent():
            self.opacityEffect.setOpacity(1)
        else:
            self.setWindowOpacity(1)

    def hide(self):
        # start hiding
        self.opacityAni.setDirection(self.opacityAni.Backward)
        self.opacityAni.setDuration(500)
        self.opacityAni.start()

    def eventFilter(self, source, event):
        if source == self.parent() and event.type() == QtCore.QEvent.Resize:
            self.opacityAni.stop()
            parentRect = self.parent().rect()
            geo = self.geometry()
            if self.corner == QtCore.Qt.TopLeftCorner:
                geo.moveTopLeft(
                    parentRect.topLeft() + QtCore.QPoint(self.margin, self.margin)
                )
            elif self.corner == QtCore.Qt.TopRightCorner:
                geo.moveTopRight(
                    parentRect.topRight() + QtCore.QPoint(-self.margin, self.margin)
                )
            elif self.corner == QtCore.Qt.BottomRightCorner:
                geo.moveBottomRight(
                    parentRect.bottomRight() + QtCore.QPoint(-self.margin, -self.margin)
                )
            else:
                geo.moveBottomLeft(
                    parentRect.bottomLeft() + QtCore.QPoint(self.margin, -self.margin)
                )
            self.setGeometry(geo)
            self.restore()
            self.timer.start()
        return super(QToaster, self).eventFilter(source, event)

    def enterEvent(self, event):
        self.restore()

    def leaveEvent(self, event):
        self.timer.start()

    def closeEvent(self, event):
        # we don't need the notification anymore, delete it!
        self.deleteLater()

    def resizeEvent(self, event):
        super(QToaster, self).resizeEvent(event)
        # if you don't set a stylesheet, you don't need any of the following!
        if not self.parent():
            # there's no parent, so we need to update the mask
            path = QtGui.QPainterPath()
            path.addRoundedRect(QtCore.QRectF(self.rect()).translated(-0.5, -0.5), 4, 4)
            self.setMask(
                QtGui.QRegion(path.toFillPolygon(QtGui.QTransform()).toPolygon())
            )
        else:
            self.clearMask()

    @staticmethod
    def showMessage(
        parent,
        message,
        icon=QtWidgets.QStyle.SP_MessageBoxInformation,
        corner=QtCore.Qt.TopLeftCorner,
        margin=10,
        closable=True,
        timeout=5000,
        desktop=False,
        parentWindow=True,
    ):

        if parent and parentWindow:
            parent = parent.window()

        if not parent or desktop:
            self = QToaster(None)
            self.setWindowFlags(
                self.windowFlags()
                | QtCore.Qt.FramelessWindowHint
                | QtCore.Qt.BypassWindowManagerHint
            )
            # This is a dirty hack!
            # parentless objects are garbage collected, so the widget will be
            # deleted as soon as the function that calls it returns, but if an
            # object is referenced to *any* other object it will not, at least
            # for PyQt (I didn't test it to a deeper level)
            self.__self = self

            currentScreen = QtWidgets.QApplication.primaryScreen()
            if parent and parent.window().geometry().size().isValid():
                # the notification is to be shown on the desktop, but there is a
                # parent that is (theoretically) visible and mapped, we'll try to
                # use its geometry as a reference to guess which desktop shows
                # most of its area; if the parent is not a top level window, use
                # that as a reference
                reference = parent.window().geometry()
            else:
                # the parent has not been mapped yet, let's use the cursor as a
                # reference for the screen
                reference = QtCore.QRect(
                    QtGui.QCursor.pos() - QtCore.QPoint(1, 1), QtCore.QSize(3, 3)
                )
            maxArea = 0
            for screen in QtWidgets.QApplication.screens():
                intersected = screen.geometry().intersected(reference)
                area = intersected.width() * intersected.height()
                if area > maxArea:
                    maxArea = area
                    currentScreen = screen
            parentRect = currentScreen.availableGeometry()
        else:
            self = QToaster(parent)
            parentRect = parent.rect()

        self.timer.setInterval(timeout)

        # use Qt standard icon pixmaps; see:
        # https://doc.qt.io/qt-5/qstyle.html#StandardPixmap-enum
        if isinstance(icon, QtWidgets.QStyle.StandardPixmap):
            labelIcon = QtWidgets.QLabel()
            self.layout().addWidget(labelIcon)
            icon = self.style().standardIcon(icon)
            size = self.style().pixelMetric(QtWidgets.QStyle.PM_SmallIconSize)
            labelIcon.setPixmap(icon.pixmap(size))

        self.label = QtWidgets.QLabel(message)
        self.layout().addWidget(self.label)

        if closable:
            self.closeButton = QtWidgets.QToolButton()
            self.layout().addWidget(self.closeButton)
            closeIcon = self.style().standardIcon(
                QtWidgets.QStyle.SP_TitleBarCloseButton
            )
            self.closeButton.setIcon(closeIcon)
            self.closeButton.setAutoRaise(True)
            self.closeButton.clicked.connect(self.close)

        self.timer.start()

        # raise the widget and adjust its size to the minimum
        self.raise_()
        self.adjustSize()

        self.corner = corner
        self.margin = margin

        geo = self.geometry()
        # now the widget should have the correct size hints, let's move it to the
        # right place
        if corner == QtCore.Qt.TopLeftCorner:
            geo.moveTopLeft(parentRect.topLeft() + QtCore.QPoint(margin, margin))
        elif corner == QtCore.Qt.TopRightCorner:
            geo.moveTopRight(parentRect.topRight() + QtCore.QPoint(-margin, margin))
        elif corner == QtCore.Qt.BottomRightCorner:
            geo.moveBottomRight(
                parentRect.bottomRight() + QtCore.QPoint(-margin, -margin)
            )
        else:
            geo.moveBottomLeft(parentRect.bottomLeft() + QtCore.QPoint(margin, -margin))

        self.setGeometry(geo)
        self.show()
        self.opacityAni.start()
