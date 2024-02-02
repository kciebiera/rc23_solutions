import mujoco
from mujoco import viewer
import numpy as np
import time


model = None
data = None
renderer = None
viewer_window = None


xml_path = "manipulator3d_ex.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
viewer_window = mujoco.viewer.launch_passive(model, data)


def position_to_xyz(position):
    position = position.round(1)
    x = position[0]
    y = position[2]
    z = position[1]
    return {"x": x, "y": y, "z": z}


def check_pos(qpos):
    data.qpos = qpos
    mujoco.mj_step(model, data)
    viewer_window.sync()
    position_Q = data.site_xpos[0]
    return position_to_xyz(position_Q)


positions = [
    [0, 0, 0],
    [0, 0.5, 0],
    [0, 0.5, np.pi / 2],
    [np.pi / 2, 0.5, np.pi / 2],
    [np.pi, 0.5, np.pi / 2],
    [np.pi, 0.5, np.pi],
]


def fk(qpos):
    data.qpos = qpos
    theta_1 = qpos[0]
    length_2 = qpos[1]
    theta_3 = qpos[2]
    position_Q = np.array(
        [
            1 / 2 * np.cos(theta_1) * np.sin(theta_3),
            1 / 2 * np.cos(theta_3) + 1 / 2 + length_2,
            1 / 2 * np.sin(theta_1) * np.sin(theta_3),
        ]
    ).round(1)

    return {"x": position_Q[0], "y": position_Q[1], "z": position_Q[2]}


for position in positions:
    print(position, check_pos(position), fk(position))
    time.sleep(1)
