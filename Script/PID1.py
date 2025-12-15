import os
import math
import mujoco
import mujoco_viewer
import numpy as np

# Portable XML path
here = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(here, "..", "Mujoco", "mod5.xml")
xml_path = os.path.normpath(xml_path)

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
dt = float(model.opt.timestep)

# Motors IDs
left_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_motor")
right_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_motor")
if left_motor_id < 0 or right_motor_id < 0:
    raise RuntimeError("Actuators left_motor/right_motor not found in mod5.xml")

# Motor limits (respect unlimited actuators)
act_ctrllim = model.actuator_ctrllimited
ctrl_min = np.full(model.nu, -np.inf)
ctrl_max = np.full(model.nu, np.inf)
for a in range(model.nu):
    if act_ctrllim[a]:
        ctrl_min[a] = model.actuator_ctrlrange[a, 0]
        ctrl_max[a] = model.actuator_ctrlrange[a, 1]
MAX_SPEED = 120.0

# IMU sensor addresses
gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
acc_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "acc")
gyro_adr = model.sensor_adr[gyro_id]
acc_adr = model.sensor_adr[acc_id]


# ---------- Get pitch angle ---------- 
pitch_est = 0.0      # filtered pitch angle
alpha = 0.98         # complementary filter coefficient
# If direction is wrong, set to -1.0
PITCH_SIGN = 1.0

def get_pitch_from_imu(data):
    """
    Данная функция:
        1) берет от гироскопа в теле корпуса угловую скорость вокруг оси Y (ось колес)
        и записывает в переменную "скорость наклона", 
        2) берет от акселерометра в теле корпуса проекцию гравитации на ось X (ось движения)
        и рассчитывает по формуле наклон относительно оси Z (вертикали)
        3) далее применяется комплементарный фильтр для оценки угла отклонения
        4) возвращаем оценку угла наклона и угловой скорости для ПИД-регулятора
    """
    global pitch_est
    gyro_y = data.sensordata[gyro_adr + 1]   # rotation around Y
    pitch_rate = gyro_y

    ax = data.sensordata[acc_adr + 0]
    az = data.sensordata[acc_adr + 2]
    # theta ≈ -atan2(ax, az) for pitch about Y
    pitch_acc = np.arctan2(-ax, az)

    pitch_gyro = pitch_est + pitch_rate * dt
    pitch_est = alpha * pitch_gyro + (1 - alpha) * pitch_acc

    return PITCH_SIGN * pitch_est


# ---------- Forward-Backward driving signal ---------- 
amp = 0.9
freq = 0.1

def drive_signal(t):
    return amp * np.sin(2 * np.pi * freq * t)

# ---------- Forward driving signal ---------- 
def drive_forward_signal():
    return 0


# ---------- Calculate PID signal ----------
Kp = 0.5
Kd = 0.1
Ki = 0.05
integral_error = 0.0
I_LIMIT = 1.0
derivative_error = 0.0
prev_error = 0.0

def calculate_pid(pitch):
    global integral_error, derivative_error, prev_error
    # If pitch>0 (forward), command positive wheel speed
    error = pitch
    integral_error += error * dt
    # anti-windup
    if integral_error > I_LIMIT:
        integral_error = I_LIMIT
    elif integral_error < -I_LIMIT:
        integral_error = -I_LIMIT
    derivative_error = (error - prev_error) / dt
    prev_error = error
    return Kp * error + Kd * derivative_error + Ki * integral_error


INIT_PITCH_DEG = 3.0  # initial forward tilt (degrees)

def place_robot_on_ground(model, data):
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        return
    lw_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wheel")
    lw_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_wheel_geom")
    if lw_body >= 0 and lw_geom >= 0:
        wheel_r = float(model.geom_size[lw_geom, 0])
        lw_local_z = float(model.body_pos[lw_body, 2])
        z0 = wheel_r - lw_local_z
    else:
        z0 = 0.12
    th = math.radians(INIT_PITCH_DEG)
    qw = math.cos(th * 0.5)
    qy = math.sin(th * 0.5)
    data.qpos[:7] = np.array([0.0, 0.0, z0, qw, 0.0, qy, 0.0], dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

# Place and settle
place_robot_on_ground(model, data)
for _ in range(5):
    mujoco.mj_step(model, data)

# ---------- Real-time simulation loop ---------- 
viewer = mujoco_viewer.MujocoViewer(model, data, title="mod5 PID1: IMU PID balancer")
try:
    while True:
        alive_attr = getattr(viewer, "is_alive")
        alive = alive_attr if isinstance(alive_attr, bool) else alive_attr()
        if not alive:
            break

        # time
        t = data.time

        # Get pitch angle
        pitch = get_pitch_from_imu(data)

        # Calculate both signals
        u_pid = calculate_pid(pitch)
        u_drive = drive_forward_signal()

        # Combine signals
        u_left = u_drive + u_pid
        u_right = u_drive + u_pid

        # Clip values to actuator limits (or fallback to MAX_SPEED)
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
