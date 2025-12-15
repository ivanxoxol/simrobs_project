import os
import math
import csv
import numpy as np
import mujoco
import mujoco_viewer


# -----------------------------
# Config (tweak as needed)
# -----------------------------

INIT_PITCH_DEG = 45.0        # initial tilt (deg); set 0.0 to start upright
MAX_SPEED = 6.0             # wheel command saturation (rad/s) for velocity actuators

# PID gains for pitch stabilization (target pitch = 0)
# Note on tuning:
# - Excessive command smoothing and rate limiting add phase lag and can induce oscillations.
#   Keep KD modestly high and reduce output filtering to maintain phase margin.
KP = 0.5
KD = 1.5
KI = 1.0
I_LIMIT = 1.0

# If pitch sign appears inverted, flip ANGLE_SIGN
ANGLE_SIGN = 1.0            # multiply measured pitch and pitch_rate
MOTOR_SIGN = 1.0            # multiply command to actuators
AUTO_MOTOR_CALIBRATE = True


def resolve_xml_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(here, "..", "Mujoco", "mod5.xml")
    return os.path.normpath(xml_path)


def place_robot_on_ground(model: mujoco.MjModel, data: mujoco.MjData, init_pitch_deg: float) -> None:
    """Set initial orientation (small pitch) and zero planar slides. Base z is fixed in XML."""
    th = math.radians(init_pitch_deg)
    qw = math.cos(th * 0.5)
    qy = math.sin(th * 0.5)

    # Support both freejoint and ball+slides models
    # Try to find a freejoint first
    if model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        data.qpos[:7] = np.array([0.0, 0.0, 0.0, qw, 0.0, qy, 0.0], dtype=float)
    else:
        # Set ball joint quaternion and zero planar slides
        j_ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_orient")
        jx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_x")
        jy = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_y")
        if j_ball >= 0:
            adr_q = model.jnt_qposadr[j_ball]
            data.qpos[adr_q:adr_q+4] = np.array([qw, 0.0, qy, 0.0], dtype=float)
        if jx >= 0:
            data.qpos[model.jnt_qposadr[jx]] = 0.0
        if jy >= 0:
            data.qpos[model.jnt_qposadr[jy]] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def get_pitch_from_quat(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Pitch (rad) around Y from base quaternion using xyz Euler convention."""
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    q = data.xquat[base_id]  # [w, x, y, z]
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    # Euler xyz: pitch around Y: asin(2*(w*y - z*x))
    sinp = 2.0 * (w * y - z * x)
    sinp = max(min(sinp, 1.0), -1.0)
    return math.asin(sinp)


def get_pitch_rate_from_gyro(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
    adr = model.sensor_adr[gyro_id]
    return float(data.sensordata[adr + 1])  # y-axis


def main():
    global MOTOR_SIGN
    xml_path = resolve_xml_path()
    print(f"Loading model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    dt = float(model.opt.timestep)

    # Resolve actuators
    left_motor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_motor")
    right_motor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_motor")
    if left_motor < 0 or right_motor < 0:
        raise RuntimeError("Actuators left_motor/right_motor not found in mod5.xml")

    # Determine control ranges (if any)
    ctrl_min = -np.inf
    ctrl_max = np.inf
    if model.actuator_ctrllimited[left_motor]:
        ctrl_min = float(model.actuator_ctrlrange[left_motor, 0])
        ctrl_max = float(model.actuator_ctrlrange[left_motor, 1])

    # Initial placement (optionally with a small tilt)
    place_robot_on_ground(model, data, INIT_PITCH_DEG)

    # Let contacts settle briefly
    for _ in range(50):
        mujoco.mj_step(model, data)

    if AUTO_MOTOR_CALIBRATE:
        # Ensure positive motor command reduces |pitch| from initial tilt
        def trial_pitch_delta(sign: float) -> float:
            place_robot_on_ground(model, data, INIT_PITCH_DEG)
            for _ in range(5):
                mujoco.mj_step(model, data)
            p0 = abs(get_pitch_from_quat(model, data))
            u_test = 0.8  # rad/s very small speed
            steps = max(40, int(0.08 / dt))
            for _ in range(steps):
                data.ctrl[left_motor] = sign * u_test
                data.ctrl[right_motor] = sign * u_test
                mujoco.mj_step(model, data)
            p1 = abs(get_pitch_from_quat(model, data))
            data.ctrl[left_motor] = 0.0
            data.ctrl[right_motor] = 0.0
            return p1 - p0  # negative is good

        try:
            d_pos = trial_pitch_delta(+1.0)
            d_neg = trial_pitch_delta(-1.0)
            MOTOR_SIGN = +1.0 if d_pos <= d_neg else -1.0
            # Reset to initial pose
            place_robot_on_ground(model, data, INIT_PITCH_DEG)
            for _ in range(10):
                mujoco.mj_step(model, data)
            print(f"Motor polarity auto-calibrated by |pitch|: MOTOR_SIGN={MOTOR_SIGN:+.1f} (d_pos={d_pos:+.4f}, d_neg={d_neg:+.4f})")
        except Exception as e:
            print(f"Auto-calibration failed, using MOTOR_SIGN={MOTOR_SIGN:+.1f}: {e}")
    else:
        print(f"Motor polarity fixed: MOTOR_SIGN={MOTOR_SIGN:+.1f}")

    # PID state
    integ = 0.0
    # Derivative low-pass on gyro (reduce noise and phase lag vs. output filtering)
    RATE_TAU = 0.03  # s, first-order LP for pitch_rate
    rate_filt = 0.0
    # First-order smoothing for commanded speed (kept modest to avoid phase lag)
    u_filt = 0.0
    TAU_U = 0.04  # s, speed command smoothing time constant (was 0.15)
    RAMP_TAU = 0.6  # s, soft start for applied speed
    # Rate limiter for wheel speed command (rad/s per step)
    DU_MAX = 0.03   # was 0.01; reduce choking to avoid limit-cycle from severe rate-limiting
    u_prev = 0.0
    sat_prev = False

    # Pre-resolve wheel joint dof for logging
    j_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hinge")
    j_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hinge")
    dof_left = model.jnt_dofadr[j_left] if j_left >= 0 else None
    dof_right = model.jnt_dofadr[j_right] if j_right >= 0 else None

    # Prepare logging
    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(here, "segway_timeseries.csv")
    png_path = os.path.join(here, "segway_timeseries.png")
    CSV_IMMEDIATE = False
    csv_file = None
    csv_writer = None
    if CSV_IMMEDIATE:
        try:
            csv_file = open(csv_path, mode="w", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["time","pitch","pitch_rate","u","u_left","u_right","wl","wr","sat"])
            csv_file.flush()
        except Exception as e:
            print(f"CSV open failed: {e}")

    times = []
    pitch_log = []
    rate_log = []
    u_log = []
    u_left_log = []
    u_right_log = []
    wl_log = []
    wr_log = []
    sat_log = []
    LOG_DECIMATE = 10

    # Viewer
    viewer = mujoco_viewer.MujocoViewer(model, data, title="Segway PID (mod5)")
    STEPS_PER_RENDER = 10
    try:
        step = 0
        while True:
            alive_attr = getattr(viewer, "is_alive")
            alive = alive_attr if isinstance(alive_attr, bool) else alive_attr()
            if not alive:
                break
            for _ in range(STEPS_PER_RENDER):
                # Measure state
                pitch = ANGLE_SIGN * get_pitch_from_quat(model, data)
                pitch_rate = ANGLE_SIGN * get_pitch_rate_from_gyro(model, data)

                # PID around zero (drive into lean): u = Kp*pitch - Kd*rate + Ki*∫error
                # Low-pass filter gyro to stabilize D-term
                alpha_rate = dt / (RATE_TAU + dt)
                rate_filt = rate_filt + alpha_rate * (pitch_rate - rate_filt)

                error = pitch
                # Form PID using current integrator (update integrator AFTER saturation handling)
                u_cmd = KP * error - KD * rate_filt + KI * integ

                alpha = dt / (TAU_U + dt)
                u_smooth = u_filt + alpha * (u_cmd - u_filt)
                u_filt = u_smooth

                ramp = 1.0 - math.exp(-data.time / RAMP_TAU)
                u_target = u_smooth * ramp
                du = u_target - u_prev
                if du > DU_MAX:
                    du = DU_MAX
                elif du < -DU_MAX:
                    du = -DU_MAX
                u_rl = u_prev + du
                u_prev = u_rl
                u_sat = float(np.clip(u_rl, -MAX_SPEED, MAX_SPEED))
                if np.isfinite(ctrl_min):
                    u_sat = float(np.clip(u_sat, ctrl_min, ctrl_max))
                saturated = 1 if abs(u_sat - u_rl) > 1e-9 else 0

                # Simple anti-windup: only integrate when not saturated,
                # or when error tends to move controller output away from saturation.
                if KI != 0.0:
                    allow_integ = (saturated == 0) or (u_rl * error < 0.0)
                    if allow_integ:
                        integ += error * dt
                        integ = max(min(integ, I_LIMIT), -I_LIMIT)
                # If integral gain is zero (default), keep integ at 0 without updating.

                u_applied = MOTOR_SIGN * u_sat
                data.ctrl[left_motor] = u_applied
                data.ctrl[right_motor] = u_applied

                mujoco.mj_step(model, data)

                # Debug and log (decimated)
                step += 1
                if step % LOG_DECIMATE == 0:
                    t = float(data.time)
                    wl = float(data.qvel[dof_left]) if dof_left is not None else 0.0
                    wr = float(data.qvel[dof_right]) if dof_right is not None else 0.0
                    times.append(t)
                    pitch_log.append(float(pitch))
                    rate_log.append(float(pitch_rate))
                    u_log.append(float(u_cmd))
                    u_left_log.append(float(u_applied))
                    u_right_log.append(float(u_applied))
                    wl_log.append(wl)
                    wr_log.append(wr)
                    sat_log.append(int(saturated))

                    if csv_writer is not None:
                        try:
                            csv_writer.writerow([t, float(pitch), float(pitch_rate), float(u_cmd), float(u_applied), float(u_applied), wl, wr, saturated])
                            csv_file.flush()
                        except Exception as e:
                            print(f"CSV write failed: {e}")

                if step % int(max(1, 0.1 / dt)) == 0:
                    print(f"t={data.time:.2f}  pitch={pitch:+.3f} rad  rate={pitch_rate:+.3f}  u={u_applied:+.2f}")

            viewer.render()
    finally:
        viewer.close()

    # Close CSV
    if csv_file is not None:
        try:
            csv_file.close()
        except Exception:
            pass

    # Plot if we have data
    if len(times) > 1:
        try:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            ax1.plot(times, pitch_log, label="pitch (rad)")
            ax1.axhline(0, color='k', lw=0.5)
            ax1.set_ylabel("angle, rad")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            ax2.plot(times, u_log, label="u_cmd", color='C3')
            ax2.plot(times, u_left_log, label="u_left", color='C2', alpha=0.7)
            ax2.set_ylabel("ctrl")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            ax3.plot(times, wl_log, label="wl (rad/s)")
            ax3.plot(times, wr_log, label="wr (rad/s)")
            ax3.step(times, sat_log, where='post', label="sat", color='C1', alpha=0.5)
            ax3.set_xlabel("time, s")
            ax3.set_ylabel("w, sat")
            ax3.grid(True, alpha=0.3)
            ax3.legend()

            fig.suptitle("Segway PID: pitch, control, wheel speeds")
            fig.tight_layout()
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            print(f"Saved plot: {png_path}")
            print(f"Saved data: {csv_path}")
        except Exception as e:
            print(f"Plotting failed: {e}")
        if not CSV_IMMEDIATE:
            try:
                import numpy as _np
                arr = _np.column_stack([
                    times, pitch_log, rate_log, u_log, u_left_log, u_right_log, wl_log, wr_log, sat_log
                ])
                _np.savetxt(
                    csv_path,
                    arr,
                    delimiter=",",
                    header="time,pitch,pitch_rate,u,u_left,u_right,wl,wr,sat",
                    comments="",
                )
                print(f"CSV saved: {csv_path}")
            except Exception as e:
                print(f"CSV save failed: {e}")


if __name__ == "__main__":
    main()
