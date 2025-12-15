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

# Ограничение по моменту (должно соответствовать ctrlrange в XML)
U_MAX = 2.5

# Знаки
ANGLE_SIGN = 1.0
MOTOR_SIGN = 1.0
AUTO_MOTOR_CALIBRATE = True

# -----------------------------
# LQR настройки
# -----------------------------
CONTROL_DECIMATE = 5  # держим u постоянным CONTROL_DECIMATE шагов

# z = [x, xd, theta, thetad]
Q = np.diag([2.0, 0.5, 80.0, 3.0])
R = np.array([[0.6]])

# Численная линеаризация
EPS_X = 1e-4
EPS_XD = 1e-4
EPS_TH = 1e-4
EPS_THD = 1e-4
EPS_U = 1e-3

# Riccati итерации
DARE_MAX_ITERS = 500
DARE_EPS = 1e-9


def project_root_from_script() -> str:
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
        w.writerow([
            "time",
            "x", "xd",
            "theta", "thetad",
            "u_cmd", "u_sat", "u_applied",
            "sat"
        ])
        w.writerows(rows)


def save_plot(png_path: str, rows: list[list[float]]) -> None:
    import matplotlib.pyplot as plt

    arr = np.array(rows, dtype=float)
    t = arr[:, 0]
    x = arr[:, 1]
    theta = arr[:, 3]
    u = arr[:, 7]  # u_applied

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax1.plot(t, theta, label="theta")
    ax1.axhline(0.0, linewidth=0.8)
    ax1.set_ylabel("theta (rad)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(t, x, label="x")
    ax2.axhline(0.0, linewidth=0.8)
    ax2.set_ylabel("x (m)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.plot(t, u, label="u_applied")
    ax3.axhline(0.0, linewidth=0.8)
    ax3.set_ylabel("u_applied")
    ax3.set_xlabel("time (s)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    fig.suptitle("segway_lqr: theta, x, control")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def dare_solve(A: np.ndarray, B: np.ndarray, Qm: np.ndarray, Rm: np.ndarray) -> np.ndarray:
    P = Qm.copy()
    for _ in range(DARE_MAX_ITERS):
        BT_P = B.T @ P
        S = Rm + BT_P @ B
        P_next = A.T @ P @ A - (A.T @ P @ B) @ np.linalg.inv(S) @ (BT_P @ A) + Qm
        if np.max(np.abs(P_next - P)) < DARE_EPS:
            return P_next
        P = P_next
    return P


def lqr_gain(A: np.ndarray, B: np.ndarray, Qm: np.ndarray, Rm: np.ndarray) -> np.ndarray:
    P = dare_solve(A, B, Qm, Rm)
    K = np.linalg.inv(Rm + B.T @ P @ B) @ (B.T @ P @ A)
    return K


def main():
    global MOTOR_SIGN

    xml_path = resolve_xml_path()
    logs_dir = resolve_logs_dir()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(logs_dir, f"segway_lqr_{ts}.csv")
    png_path = os.path.join(logs_dir, f"segway_lqr_{ts}.png")

    print(f"Loading model: {xml_path}")
    print(f"Logs dir:     {logs_dir}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)
    Ts = CONTROL_DECIMATE * dt

    left_motor = aid(model, "left_motor")
    right_motor = aid(model, "right_motor")

    # ---------- motor sign auto-calibration ----------
    set_initial_pose(model, data, INIT_PITCH_DEG)
    for _ in range(300):
        mujoco.mj_step(model, data)

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
            return th1 - th0

        d_pos = trial(+1.0)
        d_neg = trial(-1.0)
        MOTOR_SIGN = +1.0 if d_pos <= d_neg else -1.0
        print(f"MOTOR_SIGN={MOTOR_SIGN:+.1f} (d_pos={d_pos:+.4f}, d_neg={d_neg:+.4f})")

    # ---------- LQR linearization around equilibrium ----------
    base = mujoco.MjData(model)
    set_initial_pose(model, base, 0.0)
    for _ in range(500):
        mujoco.mj_step(model, base)

    jx = jid(model, "chassis_x")
    jp = jid(model, "torso_pitch")
    x_adr = model.jnt_qposadr[jx]
    th_adr = model.jnt_qposadr[jp]
    xd_dof = model.jnt_dofadr[jx]
    thd_dof = model.jnt_dofadr[jp]

    def pack_z(d: mujoco.MjData) -> np.ndarray:
        x = float(d.qpos[x_adr])
        th = float(d.qpos[th_adr])
        xd = float(d.qvel[xd_dof])
        thd = float(d.qvel[thd_dof])
        return np.array([x, xd, th, thd], dtype=float)

    def apply_z(d: mujoco.MjData, z: np.ndarray) -> None:
        d.qpos[:] = base.qpos
        d.qvel[:] = base.qvel
        d.qpos[x_adr] = z[0]
        d.qvel[xd_dof] = z[1]
        d.qpos[th_adr] = z[2]
        d.qvel[thd_dof] = z[3]
        mujoco.mj_forward(model, d)

    def step_map(z: np.ndarray, u_scalar: float) -> np.ndarray:
        d = mujoco.MjData(model)
        apply_z(d, z)
        u = float(u_scalar) * MOTOR_SIGN
        d.ctrl[left_motor] = u
        d.ctrl[right_motor] = u
        for _ in range(CONTROL_DECIMATE):
            mujoco.mj_step(model, d)
        return pack_z(d)

    z0 = np.zeros(4, dtype=float)

    A = np.zeros((4, 4), dtype=float)
    B = np.zeros((4, 1), dtype=float)

    eps = np.array([EPS_X, EPS_XD, EPS_TH, EPS_THD], dtype=float)

    for i in range(4):
        dz = np.zeros(4)
        dz[i] = eps[i]
        zp = step_map(z0 + dz, 0.0)
        zm = step_map(z0 - dz, 0.0)
        A[:, i] = (zp - zm) / (2.0 * eps[i])

    up = step_map(z0, +EPS_U)
    um = step_map(z0, -EPS_U)
    B[:, 0] = (up - um) / (2.0 * EPS_U)

    K = lqr_gain(A, B, Q, R)  # u = -K z
    print(f"LQR computed for Ts={Ts:.6f}s")
    print("K =", K)

    # ---------- run ----------
    set_initial_pose(model, data, INIT_PITCH_DEG)
    for _ in range(300):
        mujoco.mj_step(model, data)

    viewer = mujoco_viewer.MujocoViewer(model, data, title="segway_lqr")
    STEPS_PER_RENDER = 10

    rows: list[list[float]] = []
    LOG_EVERY = 10

    u_hold = 0.0
    step = 0

    try:
        while True:
            alive_attr = getattr(viewer, "is_alive")
            alive = alive_attr if isinstance(alive_attr, bool) else alive_attr()
            if not alive:
                break

            for _ in range(STEPS_PER_RENDER):
                if step % CONTROL_DECIMATE == 0:
                    x, xd, theta, thetad = get_state(model, data)
                    z = np.array([x, xd, theta, thetad], dtype=float)
                    z[2] *= ANGLE_SIGN
                    z[3] *= ANGLE_SIGN
                    u_hold = float(-(K @ z.reshape(-1, 1))[0, 0])

                u_sat = float(np.clip(u_hold, -U_MAX, U_MAX))
                sat = 1 if abs(u_sat - u_hold) > 1e-9 else 0

                u_applied = MOTOR_SIGN * u_sat
                data.ctrl[left_motor] = u_applied
                data.ctrl[right_motor] = u_applied

                mujoco.mj_step(model, data)

                step += 1
                if step % LOG_EVERY == 0:
                    x, xd, theta, thetad = get_state(model, data)
                    theta = ANGLE_SIGN * theta
                    thetad = ANGLE_SIGN * thetad
                    rows.append([
                        float(data.time),
                        float(x), float(xd),
                        float(theta), float(thetad),
                        float(u_hold), float(u_sat), float(u_applied),
                        int(sat),
                    ])

                if step % int(max(1, 0.2 / dt)) == 0:
                    x, xd, theta, thetad = get_state(model, data)
                    print(f"t={data.time:.2f} x={x:+.3f} theta={ANGLE_SIGN*theta:+.3f} u={u_applied:+.2f}")

            viewer.render()

    finally:
        viewer.close()

    if rows:
        save_csv(csv_path, rows)
        save_plot(png_path, rows)
        print(f"Saved CSV: {csv_path}")
        print(f"Saved PNG: {png_path}")
    else:
        print("No data logged (rows empty).")


if __name__ == "__main__":
    main()