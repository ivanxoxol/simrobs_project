import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("Mujoco\\mod5.xml")
data = mujoco.MjData(model)
dt = model.opt.timestep

# Motors IDs
left_motor_id = model.actuator("left_motor").id
right_motor_id = model.actuator("right_motor").id

# Motor limits
ctrl_min = model.actuator_ctrlrange[:, 0]
ctrl_max = model.actuator_ctrlrange[:, 1]

# IMU sensor addresses
gyro_id = model.sensor("gyro").id
gyro_adr = model.sensor_adr[gyro_id]
acc_id = model.sensor("acc").id
acc_adr = model.sensor_adr[acc_id]


# ---------- Get pitch angle ---------- 
pitch_est = 0.0      # filtered pitch angle
alpha = 0.98         # complementary filter coefficient

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
    pitch_acc = np.arctan2(ax, az)

    pitch_gyro = pitch_est + pitch_rate * dt
    pitch_est = alpha * pitch_gyro + (1 - alpha) * pitch_acc

    return pitch_est


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
derivative_error = 0.0
prev_error = 0.0

def calculate_pid(pitch):
    global integral_error, derivative_error, prev_error
    error = 0 - pitch
    integral_error += error * dt
    derivative_error = (error - prev_error) / dt
    prev_error = error
    return -(Kp * error + Kd * derivative_error + Ki * integral_error)


# Set initial posture strictly vertical (freejoint)
data.qpos[:] = 0
data.qvel[:] = 0
mujoco.mj_forward(model, data)

# ---------- Real-time simulation loop ---------- 
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running(): 
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

        # Clip values to actuator limits
        clipped_values = float(np.clip(u_left, ctrl_min[left_motor_id], ctrl_max[left_motor_id]))
        data.ctrl[left_motor_id] = float(np.clip(u_left, ctrl_min[left_motor_id], ctrl_max[left_motor_id]))
        data.ctrl[right_motor_id] = float(np.clip(u_right, ctrl_min[right_motor_id], ctrl_max[right_motor_id]))
        
        print(f"pitch = {pitch}, u_pid = {u_pid}, clipped_values = {clipped_values}")

        mujoco.mj_step(model, data)
        viewer.sync() 
