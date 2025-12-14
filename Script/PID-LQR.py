import mujoco
import mujoco.viewer
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
    """Return pitch angle (forward tilt) from freejoint quaternion."""
    quat = data.body("base").xquat
    if quat[0] == 0:
        return 0
    rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    angles = rotation.as_euler("xyz", degrees=False)
    #print(angles[0])
    return angles[0]


def get_pitch_dot(model, data):
    """Return angular velocity around X axis from freejoint qvel."""
    angular = data.joint("base_joint").qvel[-3:]
    #print(angular[0])
    return angular[0]


def get_wheel_velocity(model, data):
    """Return average forward linear speed from wheel hinge velocities."""
    wl = data.joint("left_hinge").qvel[0]
    wr = data.joint("right_hinge").qvel[0]
    w = (wl + wr) * 0.5
    #print(w)
    return w


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

    data.actuator("left_motor").ctrl = [u]
    data.actuator("right_motor").ctrl = [u]


# -------------------------------------------------------------------------
# MAIN SIMULATION LOOP
# -------------------------------------------------------------------------

def run_simulation(xml_path):
    global dt

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    dt = model.opt.timestep

    # Start balanced upright
    data.qpos[:] = 0
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            apply_control(model, data)

            mujoco.mj_step(model, data)
            viewer.sync()


# -------------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------------

if __name__ == "__main__":
    run_simulation("Mujoco\\mod5.xml")
