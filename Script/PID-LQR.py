import os
import math
import mujoco
import mujoco_viewer
import numpy as np
from scipy.spatial.transform import Rotation


# -------------------------------------------------------------------------
# GLOBAL SETTINGS
# -------------------------------------------------------------------------

controller_type = "LQR"   # "PID" or "LQR"
wheel_radius = 0.04       # your wheel radius
dt = None                 # will be set after loading model
v_target = 0.0            # desired forward velocity

# PID parameters
Kp = 25.0
Kd = 2.0
Ki = 0.5
integral = 0.0

# LQR parameters (example values)
LQR_K = np.array([-2.14, -0.035, 0.0, 2.236])


# -------------------------------------------------------------------------
# STATE EXTRACTION FUNCTIONS
# -------------------------------------------------------------------------

def get_pitch(model, data):
    """Return pitch angle (forward tilt about Y) from freejoint quaternion."""
    # base body has freejoint in mod5.xml
    quat = data.body("base").xquat
    if quat[0] == 0:
        return 0.0
    rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # [x,y,z,w]
    angles = rotation.as_euler("xyz", degrees=False)
    # Pitch is rotation around Y for this model
    return angles[1]


def get_pitch_dot(model, data):
    """Return angular velocity around Y from freejoint qvel."""
    angular = data.joint("base_joint").qvel[-3:]
    return angular[1]


def get_wheel_velocity(model, data):
    """Return average forward angular speed of wheels (rad/s)."""
    wl = float(data.joint("left_hinge").qvel[0])
    wr = float(data.joint("right_hinge").qvel[0])
    # In this model both hinges share the same axis; average is fine
    return 0.5 * (wl + wr)


# -------------------------------------------------------------------------
# PID CONTROLLER
# -------------------------------------------------------------------------

def pid_control(model, data):
    global integral

    pitch = get_pitch(model, data)
    pitch_dot = get_pitch_dot(model, data)

    error = -pitch
    integral += error * dt

    u = Kp * error - Kd * pitch_dot + Ki * integral

    # forward velocity correction
    v = get_wheel_velocity(model, data)
    v_err = v_target - v
    u += 4.0 * v_err

    return u


# -------------------------------------------------------------------------
# LQR CONTROLLER
# -------------------------------------------------------------------------

def lqr_control(model, data):
    pitch = -get_pitch(model, data)
    pitch_dot = get_pitch_dot(model, data)
    v = get_wheel_velocity(model, data)
    v_err = v_target - v

    x = np.array([pitch, pitch_dot, 0.0, v_err])
    u = -np.dot(LQR_K, x)
    return u


# -------------------------------------------------------------------------
# MOTOR OUTPUT
# -------------------------------------------------------------------------

def apply_control(model, data):
    """Compute control and apply to both motors."""
    if controller_type.upper() == "PID":
        u = pid_control(model, data)
    else:
        u = lqr_control(model, data)

    # Velocity actuators: target angular speed (rad/s)
    # Add a reasonable saturation for stability
    MAX_SPEED = 200.0
    u = float(np.clip(u, -MAX_SPEED, MAX_SPEED))

    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_motor")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_motor")
    data.ctrl[left_id] = u
    data.ctrl[right_id] = u


# -------------------------------------------------------------------------
# MAIN SIMULATION LOOP
# -------------------------------------------------------------------------

def run_simulation(xml_path):
    global dt

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    dt = model.opt.timestep

    # Start balanced upright, place wheels on ground
    if model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        lw_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wheel")
        lw_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_wheel_geom")
        if lw_body >= 0 and lw_geom >= 0:
            wheel_r = float(model.geom_size[lw_geom, 0])
            lw_local_z = float(model.body_pos[lw_body, 2])
            z0 = wheel_r - lw_local_z
        else:
            z0 = 0.12
        # small initial pitch (Y)
        INIT_PITCH_DEG = 3.0
        th = math.radians(INIT_PITCH_DEG)
        qw = math.cos(th * 0.5)
        qy = math.sin(th * 0.5)
        data.qpos[:7] = np.array([0.0, 0.0, z0, qw, 0.0, qy, 0.0], dtype=float)
    else:
        data.qpos[:] = 0
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    for _ in range(5):
        mujoco.mj_step(model, data)

    viewer = mujoco_viewer.MujocoViewer(model, data, title="mod5 PID/LQR")
    try:
        while True:
            alive_attr = getattr(viewer, "is_alive")
            alive = alive_attr if isinstance(alive_attr, bool) else alive_attr()
            if not alive:
                break

            apply_control(model, data)
            mujoco.mj_step(model, data)
            viewer.render()
    finally:
        viewer.close()


# -------------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------------

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(here, "..", "Mujoco", "mod5.xml")
    xml_path = os.path.normpath(xml_path)
    run_simulation(xml_path)
