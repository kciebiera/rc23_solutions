import mujoco
from mujoco import viewer
import numpy as np
import time
import imageio

queue = []

model = None
data = None
renderer = None
viewer_window = None
target = [0, -1.0, 1.1]


def start_sim():
    global model, data, viewer_window, renderer

    xml_path = "lab13-secret/manipulator3d.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    renderer = mujoco.Renderer(model, height=480, width=640)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer_window = mujoco.viewer.launch_passive(model, data)
    data.ctrl[0] = -0.2
    data.ctrl[1] = 0.2  # .00001
    data.ctrl[2] = 0.2
    for _ in range(100):
        mujoco.mj_step(model, data)
        renderer.update_scene(data)
        viewer_window.sync()


start_sim()
time.sleep(5)


def go_to(qpos):
    data.ctrl = qpos
    for i in range(10):
        mujoco.mj_step(model, data)
        renderer.update_scene(data)
        frame = renderer.render()
        if i == 0:
            queue.append(frame)
        viewer_window.sync()
        if np.linalg.norm(data.qpos - qpos) < 0.0001:
            break


while viewer_window.is_running():
    mujoco.mj_step(model, data)
    renderer.update_scene(data)
    viewer_window.sync()
    position_Q = data.site_xpos[0]

    J = np.zeros((3, 3))
    mujoco.mj_jac(model, data, J, None, position_Q, 3)

    try:
        Jinv = np.linalg.inv(J)
    except:
        print("Singular matrix")
        continue
    dX = target - position_Q
    dq = Jinv.dot(dX)
    go_to(data.qpos + 0.1 * dq)

writer = imageio.get_writer("lab13-public/video.mp4", fps=20)
for frame in queue:
    writer.append_data(frame)
writer.close()
