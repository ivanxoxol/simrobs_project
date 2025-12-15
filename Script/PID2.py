import os
import math
import mujoco
import mujoco_viewer
import csv
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

# ---- Pitch estimation ----
# Use IMU complementary filter for angle by default; can switch to kinematics
USE_IMU_ANGLE = False
    
# Complementary filter (pitch about Y)
pitch_filtered = 0.0
alpha = 0.995
# If robot still reacts in the wrong direction, flip this to -1.0
PITCH_SIGN = 1.0
last_pitch_acc = 0.0

def get_pitch_from_imu(_data):
    global pitch_filtered

    # Gyro around Y-axis (pitch rate)
    gyro_y = float(_data.sensordata[gyro_adr + 1])
    pitch_rate = gyro_y

    # Accelerometer-based pitch estimate: theta ≈ -atan2(ax, az)
    ax = float(_data.sensordata[acc_adr + 0])
    ay = float(_data.sensordata[acc_adr + 1])
    az = float(_data.sensordata[acc_adr + 2])
    # Robust pitch from accelerometer: atan2(-ax, sqrt(ay^2 + az^2))
    pitch_acc = np.arctan2(-ax, np.sqrt(ay * ay + az * az))
    # store for logging
    global last_pitch_acc
    last_pitch_acc = float(pitch_acc)

    pitch_gyro = pitch_filtered + pitch_rate * dt
    pitch_filtered = alpha * pitch_gyro + (1.0 - alpha) * pitch_acc
    return PITCH_SIGN * pitch_filtered, PITCH_SIGN * pitch_rate

def get_pitch_from_quat(_data):
    # Quaternion of base body: [w, x, y, z]
    quat = _data.xquat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")]
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    # Euler Y (pitch) from quaternion (xyz convention)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # Angular velocity around Y from gyro sensor (more robust than qvel frame issues)
    gyro_y = float(_data.sensordata[gyro_adr + 1])
    return PITCH_SIGN * pitch, PITCH_SIGN * gyro_y

def get_pitch_and_rate(_data):
    if USE_IMU_ANGLE:
        return get_pitch_from_imu(_data)
    else:
        # also set last_pitch_acc for logging to NaN when not using IMU angle
        global last_pitch_acc
        last_pitch_acc = float('nan')
        return get_pitch_from_quat(_data)


# ---- PID controller (command wheel speeds) ----
Kp = 22.0
Kd = 3.0
Ki = 0.5
integral = 0.0
I_LIMIT = 1.0
ERR_SIGN = -1.0   # if pitch>0 (forward), error = -pitch by default
MOTOR_SIGN = -1.0  # flip to +1.0 if wheels react opposite

def pid_control(pitch, pitch_rate):
    global integral
    # Map measured pitch to control error
    error = ERR_SIGN * pitch
    integral += error * dt
    # basic anti-windup
    if integral > I_LIMIT:
        integral = I_LIMIT
    elif integral < -I_LIMIT:
        integral = -I_LIMIT
    # Proper damping: subtract derivative term
    return Kp * error - Kd * pitch_rate + Ki * integral


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
# ---- Logging buffers ----
times = []
pitch_log = []
pitch_rate_log = []
pitch_acc_log = []
u_log = []
u_left_log = []
u_right_log = []
sat_left_log = []
sat_right_log = []
wl_log = []
wr_log = []
LOG_DECIMATE = 10
step_count = 0

# pre-resolve wheel joint qvel addresses for logging
left_hinge_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hinge")
right_hinge_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hinge")
left_hinge_dof = model.jnt_dofadr[left_hinge_id] if left_hinge_id >= 0 else None
right_hinge_dof = model.jnt_dofadr[right_hinge_id] if right_hinge_id >= 0 else None

# Prepare log file paths and open CSV immediately
here_script = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(here_script, "pid2_timeseries.csv")
png_path = os.path.join(here_script, "pid2_timeseries.png")

CSV_IMMEDIATE_WRITE = True
csv_file = None
csv_writer = None
if CSV_IMMEDIATE_WRITE:
    try:
        csv_file = open(csv_path, mode="w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["time","pitch","pitch_rate","pitch_acc","u","u_left","u_right","wl","wr","satL","satR"])
        csv_file.flush()
    except Exception as e:
        print(f"CSV open failed: {e}")

# --- Optional quick auto-polarity calibration ---
def auto_calibrate_polarity(steps=300, u_mag=8.0):
    global MOTOR_SIGN, integral
    # Try to reduce current tilt with a small fixed control
    p0, _ = get_pitch_and_rate(data)
    direction = -np.sign(p0)  # drive to reduce |pitch|
    before = abs(p0)
    for _ in range(steps):
        u_test = direction * u_mag
        if np.isfinite(ctrl_min[left_motor_id]):
            u_left = float(np.clip(u_test, ctrl_min[left_motor_id], ctrl_max[left_motor_id]))
        else:
            u_left = float(np.clip(u_test, -MAX_SPEED, MAX_SPEED))
        if np.isfinite(ctrl_min[right_motor_id]):
            u_right = float(np.clip(u_test, ctrl_min[right_motor_id], ctrl_max[right_motor_id]))
        else:
            u_right = float(np.clip(u_test, -MAX_SPEED, MAX_SPEED))
        data.ctrl[left_motor_id] = u_left
        data.ctrl[right_motor_id] = u_right
        mujoco.mj_step(model, data)
    p1, _ = get_pitch_and_rate(data)
    after = abs(p1)
    # If |pitch| grew, flip motor sign
    if after > before:
        MOTOR_SIGN *= -1.0
    # reset integrator and controls
    integral = 0.0
    data.ctrl[left_motor_id] = 0.0
    data.ctrl[right_motor_id] = 0.0

# Run auto calibration once
auto_calibrate_polarity()
try:
    while True:
        alive_attr = getattr(viewer, "is_alive")
        alive = alive_attr if isinstance(alive_attr, bool) else alive_attr()
        if not alive:
            break

        # Compute control before integration step
        pitch, pitch_rate = get_pitch_and_rate(data)

        # Balance command (wheel angular speed, rad/s)
        u_balance = pid_control(pitch, pitch_rate)
        u_drive = 0.0  # optional forward bias (rad/s)

        u_left_raw = u_drive + u_balance
        u_right_raw = u_drive + u_balance

        # Apply control with proper saturation
        if np.isfinite(ctrl_min[left_motor_id]):
            u_left = float(np.clip(u_left_raw, ctrl_min[left_motor_id], ctrl_max[left_motor_id]))
        else:
            u_left = float(np.clip(u_left_raw, -MAX_SPEED, MAX_SPEED))

        if np.isfinite(ctrl_min[right_motor_id]):
            u_right = float(np.clip(u_right_raw, ctrl_min[right_motor_id], ctrl_max[right_motor_id]))
        else:
            u_right = float(np.clip(u_right_raw, -MAX_SPEED, MAX_SPEED))

        data.ctrl[left_motor_id] = MOTOR_SIGN * u_left
        data.ctrl[right_motor_id] = MOTOR_SIGN * u_right

        # step sim
        mujoco.mj_step(model, data)

        # logging (decimated)
        step_count += 1
        if step_count % LOG_DECIMATE == 0:
            t = float(data.time)
            wl = float(data.qvel[left_hinge_dof]) if left_hinge_dof is not None else 0.0
            wr = float(data.qvel[right_hinge_dof]) if right_hinge_dof is not None else 0.0
            times.append(t)
            pitch_log.append(float(pitch))
            pitch_rate_log.append(float(pitch_rate))
            pitch_acc_log.append(float(last_pitch_acc))
            u_log.append(float(u_balance))
            u_left_log.append(float(u_left))
            u_right_log.append(float(u_right))
            wl_log.append(wl)
            wr_log.append(wr)
            sat_left_log.append(1 if u_left != u_left_raw else 0)
            sat_right_log.append(1 if u_right != u_right_raw else 0)

            if csv_writer is not None:
                try:
                    csv_writer.writerow([t, float(pitch), float(pitch_rate), float(last_pitch_acc), float(u_balance), float(u_left), float(u_right), wl, wr, 1 if u_left != u_left_raw else 0, 1 if u_right != u_right_raw else 0])
                    csv_file.flush()
                except Exception as e:
                    print(f"CSV write failed: {e}")
        viewer.render()
finally:
    viewer.close()

# Save logs if any
if csv_file is not None:
    try:
        csv_file.close()
    except Exception:
        pass

if len(times) > 0:

    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        ax1.plot(times, pitch_log, label="pitch (rad)")
        ax1.plot(times, pitch_acc_log, '--', alpha=0.6, label="pitch_acc (rad)")
        ax1.axhline(0, color='k', lw=0.5)
        ax1.set_ylabel("angle, rad")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(times, u_log, color='C3', label="u_balance (rad/s)")
        ax2.plot(times, u_left_log, color='C2', alpha=0.7, label="u_left (cmd)")
        ax2.plot(times, u_right_log, color='C1', alpha=0.7, label="u_right (cmd)")
        ax2.set_ylabel("ctrl")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        ax3.plot(times, wl_log, label="wl (rad/s)")
        ax3.plot(times, wr_log, label="wr (rad/s)")
        ax3.step(times, sat_left_log, where='post', color='C2', alpha=0.4, label="sat L")
        ax3.step(times, sat_right_log, where='post', color='C1', alpha=0.4, label="sat R")
        ax3.set_xlabel("time, s")
        ax3.set_ylabel("w, sat")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        fig.suptitle("PID2: pitch, control, wheel speeds")
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Plotting failed: {e}")

    if not CSV_IMMEDIATE_WRITE:
        try:
            import numpy as _np
            arr = _np.column_stack([
                times, pitch_log, pitch_rate_log, pitch_acc_log,
                u_log, u_left_log, u_right_log,
                wl_log, wr_log, sat_left_log, sat_right_log
            ])
            _np.savetxt(
                csv_path,
                arr,
                delimiter=",",
                header="time,pitch,pitch_rate,pitch_acc,u,u_left,u_right,wl,wr,satL,satR",
                comments="",
            )
        except Exception as e:
            print(f"CSV save failed: {e}")
    print(f"Saved plot: {png_path}")
    print(f"Saved data: {csv_path}")
