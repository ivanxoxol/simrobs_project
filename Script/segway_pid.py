import os
import math
import csv
from datetime import datetime

import numpy as np
import mujoco
import mujoco_viewer


# -----------------------------
# Настройки
# -----------------------------
XML_NAME = "segway.xml"
INIT_PITCH_DEG = 8.0

# ===== PID по УГЛУ (theta) =====
KP_THETA = 85.0
KI_THETA = 35.0
KD_THETA = 12.0
I_LIMIT = 2.0          # ограничение интегратора (рад*с)

# ===== PD по ПОЛОЖЕНИЮ тележки (чтобы не уезжал) =====
K_X = 0.8
K_XD = 1.8

# Ограничение по моменту (должно соответствовать ctrlrange в XML)
U_MAX = 2.5

# Фильтры
RATE_TAU = 0.015       # фильтр thetad (с)
U_TAU = 0.010          # сглаживание u (с)

# Знаки (на случай инверсий)
ANGLE_SIGN = 1.0
MOTOR_SIGN = 1.0
AUTO_MOTOR_CALIBRATE = True


def project_root_from_script() -> str:
    """Script/segway_pid.py -> project root"""
    here = os.path.dirname(os.path.abspath(__file__))  # .../simrobs_project/Script
    return os.path.normpath(os.path.join(here, ".."))  # .../simrobs_project


def resolve_xml_path() -> str:
    root = project_root_from_script()
    return os.path.join(root, "Mujoco", XML_NAME)


def resolve_logs_dir() -> str:
    root = project_root_from_script()
    logs_dir = os.path.join(root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def jid(model, name: str) -> int:
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if j < 0:
        raise RuntimeError(f"Joint '{name}' not found in XML")
    return j


def aid(model, name: str) -> int:
    a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if a < 0:
        raise RuntimeError(f"Actuator '{name}' not found in XML")
    return a


def set_initial_pose(model: mujoco.MjModel, data: mujoco.MjData, pitch_deg: float) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    th = math.radians(pitch_deg)

    jx = jid(model, "chassis_x")
    jp = jid(model, "torso_pitch")

    data.qpos[model.jnt_qposadr[jx]] = 0.0
    data.qpos[model.jnt_qposadr[jp]] = th

    mujoco.mj_forward(model, data)


def get_state(model: mujoco.MjModel, data: mujoco.MjData):
    jx = jid(model, "chassis_x")
    jp = jid(model, "torso_pitch")

    x = float(data.qpos[model.jnt_qposadr[jx]])
    theta = float(data.qpos[model.jnt_qposadr[jp]])

    xd = float(data.qvel[model.jnt_dofadr[jx]])
    thetad = float(data.qvel[model.jnt_dofadr[jp]])

    return x, xd, theta, thetad


def save_csv(csv_path: str, rows: list[list[float]]) -> None:
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "x", "xd", "theta", "thetad", "theta_f", "integ_theta", "u_cmd", "u_sat", "u_applied"])
        w.writerows(rows)


def save_plot(png_path: str, rows: list[list[float]]) -> None:
    # rows: time, x, xd, theta, thetad, theta_f, integ, u_cmd, u_sat, u_applied
    import matplotlib.pyplot as plt

    arr = np.array(rows, dtype=float)
    t = arr[:, 0]
    x = arr[:, 1]
    theta = arr[:, 3]
    u = arr[:, 9]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax1.plot(t, theta)
    ax1.axhline(0.0, linewidth=0.8)
    ax1.set_ylabel("theta (rad)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, x)
    ax2.axhline(0.0, linewidth=0.8)
    ax2.set_ylabel("x (m)")
    ax2.grid(True, alpha=0.3)

    ax3.plot(t, u)
    ax3.axhline(0.0, linewidth=0.8)
    ax3.set_ylabel("u_applied")
    ax3.set_xlabel("time (s)")
    ax3.grid(True, alpha=0.3)

    fig.suptitle("Segway: theta, x, control")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def main():
    global MOTOR_SIGN

    xml_path = resolve_xml_path()
    logs_dir = resolve_logs_dir()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(logs_dir, f"segway_{ts}.csv")
    png_path = os.path.join(logs_dir, f"segway_{ts}.png")

    print(f"Loading model: {xml_path}")
    print(f"Logs dir:     {logs_dir}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)

    left_motor = aid(model, "left_motor")
    right_motor = aid(model, "right_motor")

    # Стартовая поза
    set_initial_pose(model, data, INIT_PITCH_DEG)

    # Устаканить контакт
    for _ in range(300):
        mujoco.mj_step(model, data)

    # Автокалибровка направления моторов: какой знак уменьшает |theta|
    if AUTO_MOTOR_CALIBRATE:
        def trial(sign: float) -> float:
            set_initial_pose(model, data, INIT_PITCH_DEG)
            for _ in range(200):
                mujoco.mj_step(model, data)

            _, _, th0, _ = get_state(model, data)
            th0 = abs(th0)

            u_test = 0.8
            steps = max(200, int(0.25 / dt))
            for _ in range(steps):
                u = sign * u_test
                data.ctrl[left_motor] = u
                data.ctrl[right_motor] = u
                mujoco.mj_step(model, data)

            _, _, th1, _ = get_state(model, data)
            th1 = abs(th1)

            data.ctrl[left_motor] = 0.0
            data.ctrl[right_motor] = 0.0
            return th1 - th0  # отрицательно = лучше

        d_pos = trial(+1.0)
        d_neg = trial(-1.0)
        MOTOR_SIGN = +1.0 if d_pos <= d_neg else -1.0
        print(f"MOTOR_SIGN={MOTOR_SIGN:+.1f} (d_pos={d_pos:+.4f}, d_neg={d_neg:+.4f})")

        set_initial_pose(model, data, INIT_PITCH_DEG)
        for _ in range(200):
            mujoco.mj_step(model, data)

    viewer = mujoco_viewer.MujocoViewer(model, data, title="Segway PID (theta) + PD (x)")
    STEPS_PER_RENDER = 10

    # Состояния PID
    integ_theta = 0.0
    thetad_f = 0.0
    u_f = 0.0

    # Логи (децимация)
    rows: list[list[float]] = []
    LOG_EVERY = 10  # логируем каждые N шагов

    step = 0
    try:
        while True:
            alive_attr = getattr(viewer, "is_alive")
            alive = alive_attr if isinstance(alive_attr, bool) else alive_attr()
            if not alive:
                break

            for _ in range(STEPS_PER_RENDER):
                x, xd, theta, thetad = get_state(model, data)

                # цель: theta_ref = 0, x_ref = 0
                theta = ANGLE_SIGN * theta
                thetad = ANGLE_SIGN * thetad

                # фильтр D по углу
                a_rate = dt / (RATE_TAU + dt)
                thetad_f += a_rate * (thetad - thetad_f)

                # ----- PID по углу -----
                err_theta = theta
                near_upright = abs(err_theta) < 0.35

                if near_upright:
                    integ_theta += err_theta * dt
                    integ_theta = float(np.clip(integ_theta, -I_LIMIT, I_LIMIT))

                u_pid = -(KP_THETA * err_theta + KI_THETA * integ_theta + KD_THETA * thetad_f)

                # ----- PD по положению -----
                u_pos = -(K_X * x + K_XD * xd)

                u_cmd = u_pid + u_pos

                # сгладим u
                a_u = dt / (U_TAU + dt)
                u_f += a_u * (u_cmd - u_f)

                # saturation
                u_sat = float(np.clip(u_f, -U_MAX, U_MAX))

                # anti-windup: если в насыщении и интегратор толкает в ту же сторону
                if abs(u_sat - u_f) > 1e-9 and np.sign(u_f) == np.sign(err_theta):
                    integ_theta *= 0.995

                u_applied = MOTOR_SIGN * u_sat
                data.ctrl[left_motor] = u_applied
                data.ctrl[right_motor] = u_applied

                mujoco.mj_step(model, data)

                step += 1
                if step % LOG_EVERY == 0:
                    rows.append([
                        float(data.time),
                        float(x),
                        float(xd),
                        float(theta),
                        float(thetad),
                        float(thetad_f),
                        float(integ_theta),
                        float(u_cmd),
                        float(u_sat),
                        float(u_applied),
                    ])

                if step % int(max(1, 0.2 / dt)) == 0:
                    print(f"t={data.time:.2f} x={x:+.3f} theta={theta:+.3f} u={u_applied:+.2f}")

            viewer.render()

    finally:
        viewer.close()

    # Сохранение логов после закрытия окна
    if rows:
        save_csv(csv_path, rows)
        save_plot(png_path, rows)
        print(f"Saved CSV: {csv_path}")
        print(f"Saved PNG: {png_path}")
    else:
        print("No data logged (rows empty).")


if __name__ == "__main__":
    main()