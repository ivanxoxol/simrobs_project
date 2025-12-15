# simrobs_project

Two-wheel balancer and related MuJoCo experiments.

Quick start (mod5, MuJoCo 3.x viewer):
- Run `python Script/PID2.py` for complementary-filter PID (velocity actuators).
- Or run `python Script/PID-LQR.py` to try the LQR/PID variant (fixed axes).
- Recommended fresh controller: `python Script/segway_pid.py` — clean PID that
  stabilizes the Segway (mod5) at pitch=0 using quaternion pitch + gyro rate.

Notes about fixes made:
- Use portable paths to `Mujoco/mod5.xml` (not Windows backslashes).
- Respect actuator control limits; if unlimited, fall back to sane velocity limits.
- Compute pitch about the Y axis (matches wheel hinge axis in `mod5.xml`).
- Initialize the freejoint quaternion to identity `[1, 0, 0, 0]`.

Model files:
- `Mujoco/mod5.xml` — recommended physics, velocity actuators (`left_motor`, `right_motor`).
- `Mujoco/mod4.xml` — older torque-motor variant.
