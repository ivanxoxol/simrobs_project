import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("Mujoco\\mod5.xml")
data = mujoco.MjData(model)

dt = model.opt.timestep

# Actuators
left_motor_id  = model.actuator("left_motor").id
right_motor_id = model.actuator("right_motor").id

# Motor limits
ctrl_min = model.actuator_ctrlrange[:, 0]
ctrl_max = model.actuator_ctrlrange[:, 1]

# Sensor addresses
gyro_id = model.sensor("gyro").id
gyro_adr = model.sensor_adr[gyro_id]

acc_id = model.sensor("acc").id
acc_adr = model.sensor_adr[acc_id]

# ---- Complementary filter ----
pitch_filtered = 0
alpha = 0.995

def get_pitch_from_imu(data):
    global pitch_filtered

    # gyro around y-axis
    gyro_y = data.sensordata[gyro_adr + 1]
    pitch_rate = gyro_y

    # accelerometer estimate
    ax = data.sensordata[acc_adr + 0]
    az = data.sensordata[acc_adr + 2]
    pitch_acc = np.arctan2(-ax, az)

    pitch_gyro = pitch_filtered + pitch_rate * dt
    pitch_filtered = alpha * pitch_gyro + (1 - alpha) * pitch_acc

    return pitch_filtered, pitch_rate


# ---- PID controller ----
Kp = 28
Kd = 1.8
Ki = 6.0
integral = 0.0

def pid_control(pitch, pitch_rate):
    global integral
    error = -pitch
    integral += error * dt
    return Kp*error - Kd*pitch_rate + Ki*integral


# ==== Initial state ====
data.qpos[:] = 0
data.qvel[:] = 0
mujoco.mj_forward(model, data)


# ===== Simulation =====
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        pitch, pitch_rate = get_pitch_from_imu(data)

        u_balance = pid_control(pitch, pitch_rate)
        u_drive = 1.2  # target speed in rad/s

        u_left  = u_drive + u_balance
        u_right = u_drive + u_balance

        data.ctrl[left_motor_id]  = float(np.clip(u_left, ctrl_min[left_motor_id], ctrl_max[left_motor_id]))
        data.ctrl[right_motor_id] = float(np.clip(u_right, ctrl_min[right_motor_id], ctrl_max[right_motor_id]))
        
        print(f"pitch = {pitch}, u_balance = {u_balance}")

        mujoco.mj_step(model, data)
        viewer.sync()
