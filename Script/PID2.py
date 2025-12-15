import os
import math
import mujoco
import mujoco_viewer
import numpy as np

# Resolve XML path portably
here = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(here, "..", "Mujoco", "mod5.xml")
xml_path = os.path.normpath(xml_path)

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

dt = float(model.opt.timestep)

# Actuator IDs (use standard name->id for robustness)
left_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_motor")
right_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_motor")
if left_motor_id < 0 or right_motor_id < 0:
    raise RuntimeError("Actuators left_motor/right_motor not found in mod5.xml")

# Control ranges (handle unlimited actuators)
act_ctrllim = model.actuator_ctrllimited
ctrl_min = np.full(model.nu, -np.inf)
ctrl_max = np.full(model.nu, np.inf)
for a in range(model.nu):
    if act_ctrllim[a]:
        ctrl_min[a] = model.actuator_ctrlrange[a, 0]
        ctrl_max[a] = model.actuator_ctrlrange[a, 1]

# Fallback saturation for velocity actuators if unlimited
MAX_SPEED = 120.0  # rad/s

# Sensor addresses
gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
acc_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "acc")
if gyro_id < 0 or acc_id < 0:
    raise RuntimeError("Sensors gyro/acc not found in mod5.xml")
gyro_adr = model.sensor_adr[gyro_id]
acc_adr = model.sensor_adr[acc_id]

# ---- Complementary filter (pitch about Y) ----
pitch_filtered = 0.0
alpha = 0.995
# If robot still reacts in the wrong direction, flip this to -1.0
PITCH_SIGN = 1.0

def get_pitch_from_imu(_data):
    global pitch_filtered

    # Gyro around Y-axis (pitch rate)
    gyro_y = float(_data.sensordata[gyro_adr + 1])
    pitch_rate = gyro_y

    # Accelerometer-based pitch estimate: theta ≈ -atan2(ax, az)
    ax = float(_data.sensordata[acc_adr + 0])
    az = float(_data.sensordata[acc_adr + 2])
    pitch_acc = np.arctan2(-ax, az)

    pitch_gyro = pitch_filtered + pitch_rate * dt
    pitch_filtered = alpha * pitch_gyro + (1.0 - alpha) * pitch_acc
    return PITCH_SIGN * pitch_filtered, PITCH_SIGN * pitch_rate


# ---- PID controller (command wheel speeds) ----
Kp = 28.0
Kd = 1.8
Ki = 6.0
integral = 0.0
I_LIMIT = 1.0

def pid_control(pitch, pitch_rate):
    global integral
    # For this model: if pitch>0 (falls forward), drive forward (u>0)
    error = pitch
    integral += error * dt
    # basic anti-windup
    if integral > I_LIMIT:
        integral = I_LIMIT
    elif integral < -I_LIMIT:
        integral = -I_LIMIT
    return Kp * error + Kd * pitch_rate + Ki * integral


INIT_PITCH_DEG = 3.0  # initial forward tilt (degrees)

def place_robot_on_ground(model, data):
    """Place base so wheels just touch the floor (z=0)."""
    base_free = (model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE)
    if not base_free:
        return

    lw_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wheel")
    lw_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_wheel_geom")
    if lw_body < 0 or lw_geom < 0:
        # fallback to nominal 0.12 m
        z0 = 0.12
    else:
        wheel_r = float(model.geom_size[lw_geom, 0])
        lw_local_z = float(model.body_pos[lw_body, 2])
        z0 = wheel_r - lw_local_z

    # Orientation: small pitch around Y
    th = math.radians(INIT_PITCH_DEG)
    qw = math.cos(th * 0.5)
    qy = math.sin(th * 0.5)
    # qpos: [x,y,z, qw,qx,qy,qz]
    data.qpos[:7] = np.array([0.0, 0.0, z0, qw, 0.0, qy, 0.0], dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


# ==== Initial state: place on ground and settle ====
place_robot_on_ground(model, data)
for _ in range(5):
    mujoco.mj_step(model, data)


# ===== Simulation (mujoco-python-viewer) =====
viewer = mujoco_viewer.MujocoViewer(model, data, title="mod5 PID2: IMU PID balancer")
try:
    while True:
        alive_attr = getattr(viewer, "is_alive")
        alive = alive_attr if isinstance(alive_attr, bool) else alive_attr()
        if not alive:
            break

        # Compute control before integration step
        pitch, pitch_rate = get_pitch_from_imu(data)

        # Balance command (wheel angular speed, rad/s)
        u_balance = pid_control(pitch, pitch_rate)
        u_drive = 0.0  # optional forward bias (rad/s)

        u_left = u_drive + u_balance
        u_right = u_drive + u_balance

        # Apply control with proper saturation
        if np.isfinite(ctrl_min[left_motor_id]):
            u_left = float(np.clip(u_left, ctrl_min[left_motor_id], ctrl_max[left_motor_id]))
        else:
            u_left = float(np.clip(u_left, -MAX_SPEED, MAX_SPEED))

        if np.isfinite(ctrl_min[right_motor_id]):
            u_right = float(np.clip(u_right, ctrl_min[right_motor_id], ctrl_max[right_motor_id]))
        else:
            u_right = float(np.clip(u_right, -MAX_SPEED, MAX_SPEED))

        data.ctrl[left_motor_id] = u_left
        data.ctrl[right_motor_id] = u_right

        mujoco.mj_step(model, data)
        viewer.render()
finally:
    viewer.close()
