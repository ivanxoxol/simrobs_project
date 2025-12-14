import time
import numpy as np
import mujoco
import mujoco.viewer

paused = False


def key_callback(keycode):
    global paused
    if keycode == 32:  # Space key
        paused = not paused


class BipedalRobotController:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        # PID параметры для суставов
        self.pid_params = {
            'hip': {'kp': 150.0, 'ki': 0.5, 'kd': 15.0, 'integral': 0, 'prev_error': 0, 'integral_limit': 2.0},
            'knee': {'kp': 120.0, 'ki': 0.3, 'kd': 12.0, 'integral': 0, 'prev_error': 0, 'integral_limit': 1.5}
        }

        # Целевые углы для вертикальной стойки
        self.target_angles = {
            'hip': 0.0,  # радианы
            'knee': -0.25  # радианы (отрицательное значение для сгиба)
        }

        # Для управления балансом
        self.target_torso_angle = 0.0  # вертикальное положение

        # Фильтр для угла наклона
        self.pitch_filter = 0.0
        self.filter_alpha = 0.1

        # Флаг начала работы
        self.startup_phase = True
        self.startup_timer = 0

        # Для управления движением
        self.target_velocity = 0.0
        self.current_velocity = 0.0
        self.velocity_filter = 0.0

    def get_joint_angle(self, joint_name):
        """Получить угол сустава по имени"""
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            return self.data.qpos[joint_id]
        return 0.0

    def quaternion_to_euler(self, quat):
        """Преобразование кватерниона в углы Эйлера"""
        w, x, y, z = quat

        # pitch (x-вращение)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        return pitch

    def get_torso_pitch(self):
        """Получить угол наклона торса (pitch)"""
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        if torso_id >= 0:
            quat = self.data.xquat[torso_id]
            pitch = self.quaternion_to_euler(quat)

            # Применяем фильтр низких частот
            self.pitch_filter = self.pitch_filter * (1 - self.filter_alpha) + pitch * self.filter_alpha
            return self.pitch_filter
        return 0.0

    def get_torso_height(self):
        """Получить высоту торса"""
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        if torso_id >= 0:
            return self.data.xpos[torso_id][2]
        return 0.0

    def get_wheel_velocity(self):
        """Получить среднюю скорость колес"""
        left_wheel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'left_wheel_joint')
        right_wheel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'right_wheel_joint')

        if left_wheel_id >= 0 and right_wheel_id >= 0:
            left_vel = self.data.qvel[left_wheel_id]
            right_vel = self.data.qvel[right_wheel_id]
            return (left_vel + right_vel) * 0.1  # умножаем на радиус колеса
        return 0.0

    def pid_control(self, current, target, pid_params, dt):
        """Улучшенный PID контроллер"""
        error = target - current

        # Пропорциональная составляющая
        p = pid_params['kp'] * error

        # Интегральная составляющая с насыщением
        pid_params['integral'] += error * dt
        pid_params['integral'] = np.clip(
            pid_params['integral'],
            -pid_params['integral_limit'],
            pid_params['integral_limit']
        )
        i = pid_params['ki'] * pid_params['integral']

        # Дифференциальная составляющая с фильтрацией
        if dt > 0:
            error_derivative = (error - pid_params['prev_error']) / dt
            # Фильтр для производной
            error_derivative = pid_params.get('derivative_filter', 0.0) * 0.9 + error_derivative * 0.1
            pid_params['derivative_filter'] = error_derivative
            d = pid_params['kd'] * error_derivative
        else:
            d = 0

        pid_params['prev_error'] = error

        # Ограничение общего сигнала
        output = p + i + d
        return np.clip(output, -3.0, 3.0)

    def apply_control(self):
        """Применить управление ко всем суставам и колесам"""
        dt = self.model.opt.timestep

        # Плавный старт
        if self.startup_phase:
            self.startup_timer += dt
            if self.startup_timer < 0.5:  # 0.5 секунды на старт
                startup_factor = self.startup_timer / 0.5
            else:
                startup_factor = 1.0
                self.startup_phase = False
        else:
            startup_factor = 1.0

        # Получаем текущие значения
        torso_pitch = self.get_torso_pitch()
        torso_height = self.get_torso_height()
        wheel_velocity = self.get_wheel_velocity()

        # Фильтрация скорости
        self.velocity_filter = self.velocity_filter * 0.9 + wheel_velocity * 0.1
        self.current_velocity = self.velocity_filter

        # Управление балансом на основе угла наклона
        balance_gain = 0.8
        balance_correction = -torso_pitch * balance_gain * startup_factor

        # Динамическое изменение целевых углов при наклоне
        if abs(torso_pitch) > 0.3:  # Если сильный наклон
            hip_target_offset = torso_pitch * 0.4
            knee_target_offset = torso_pitch * 0.2
        else:
            hip_target_offset = balance_correction
            knee_target_offset = balance_correction * 0.5

        # Левая нога
        left_hip_angle = self.get_joint_angle('left_hip_joint')
        left_knee_angle = self.get_joint_angle('left_knee_joint')

        left_hip_target = self.target_angles['hip'] + hip_target_offset
        left_knee_target = self.target_angles['knee'] + knee_target_offset

        # Вычисляем управляющие сигналы
        left_hip_control = self.pid_control(
            left_hip_angle,
            left_hip_target,
            self.pid_params['hip'],
            dt
        )

        left_knee_control = self.pid_control(
            left_knee_angle,
            left_knee_target,
            self.pid_params['knee'],
            dt
        )

        # Правая нога
        right_hip_angle = self.get_joint_angle('right_hip_joint')
        right_knee_angle = self.get_joint_angle('right_knee_joint')

        right_hip_target = self.target_angles['hip'] + hip_target_offset
        right_knee_target = self.target_angles['knee'] + knee_target_offset

        right_hip_control = self.pid_control(
            right_hip_angle,
            right_hip_target,
            self.pid_params['hip'],
            dt
        )

        right_knee_control = self.pid_control(
            right_knee_angle,
            right_knee_target,
            self.pid_params['knee'],
            dt
        )

        # Управление колесами
        # Компенсация падения
        fall_compensation = torso_pitch * 25.0 * startup_factor

        # Управление скоростью (простейший P-регулятор)
        velocity_error = self.target_velocity - self.current_velocity
        velocity_control = velocity_error * 0.5

        # Суммарное управление колесами
        wheel_control = fall_compensation + velocity_control

        # Применяем управление к приводам
        actuator_ids = {}
        for i in range(self.model.nu):
            act_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if act_name:
                actuator_ids[act_name] = i

        # Применение управляющих сигналов
        control_scale = 0.1 * startup_factor

        if 'left_hip_motor' in actuator_ids:
            self.data.ctrl[actuator_ids['left_hip_motor']] = np.clip(left_hip_control * control_scale, -1, 1)
        if 'left_knee_motor' in actuator_ids:
            self.data.ctrl[actuator_ids['left_knee_motor']] = np.clip(left_knee_control * control_scale, -1, 1)
        if 'right_hip_motor' in actuator_ids:
            self.data.ctrl[actuator_ids['right_hip_motor']] = np.clip(right_hip_control * control_scale, -1, 1)
        if 'right_knee_motor' in actuator_ids:
            self.data.ctrl[actuator_ids['right_knee_motor']] = np.clip(right_knee_control * control_scale, -1, 1)

        # Управление колесами
        if 'left_wheel_velocity' in actuator_ids:
            self.data.ctrl[actuator_ids['left_wheel_velocity']] = np.clip(wheel_control, -10, 10)
        if 'right_wheel_velocity' in actuator_ids:
            self.data.ctrl[actuator_ids['right_wheel_velocity']] = np.clip(wheel_control, -10, 10)

        # Отладочная информация
        if hasattr(self, 'debug_counter'):
            self.debug_counter += 1
            if self.debug_counter % 200 == 0:
                print(f"Torso: pitch={np.degrees(torso_pitch):.1f}°, height={torso_height:.3f}m")
                print(f"Velocity: {self.current_velocity:.3f} m/s")
                print(f"Balance correction: {balance_correction:.3f}")
                print(f"Wheel control: {wheel_control:.3f}")
                print("-" * 40)
        else:
            self.debug_counter = 0


def main():
    global paused

    # Загружаем модель
    try:
        m = mujoco.MjModel.from_xml_path("Mujoco\\mod5.xml")
        d = mujoco.MjData(m)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    # Создаем контроллер
    controller = BipedalRobotController(m, d)

    # Выводим информацию о модели
    print(f"Модель загружена. Суставы: {m.njnt}, Приводы: {m.nu}")
    print("Список суставов:")
    for i in range(m.njnt):
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        if joint_name:
            print(f"  {i}: {joint_name}")

    print("\nСписок приводов:")
    for i in range(m.nu):
        actuator_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if actuator_name:
            print(f"  {i}: {actuator_name}")

    print("\nУправление:")
    print("  Пробел - пауза/продолжить")
    print("  ESC - выход")
    print("\nЗапуск симуляции...")

    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        # Инициализация начальной позиции
        # Устанавливаем робота в вертикальное положение
        try:
            # Находим корневой сустав
            root_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'root')
            if root_id >= 0:
                # Позиция
                d.qpos[root_id:root_id + 3] = [0, 0, 0.66]
                # Ориентация (кватернион для вертикального положения)
                d.qpos[root_id + 3:root_id + 7] = [1, 0, 0, 0]

            # Начальное сгибание коленей
            left_knee_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'left_knee_joint')
            right_knee_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'right_knee_joint')
            if left_knee_id >= 0:
                d.qpos[left_knee_id] = -0.25
            if right_knee_id >= 0:
                d.qpos[right_knee_id] = -0.25

            # Начальное положение бедер
            left_hip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'left_hip_joint')
            right_hip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'right_hip_joint')
            if left_hip_id >= 0:
                d.qpos[left_hip_id] = 0.0
            if right_hip_id >= 0:
                d.qpos[right_hip_id] = 0.0

        except Exception as e:
            print(f"Ошибка инициализации: {e}")

        step_count = 0
        start_time = time.time()

        # Запускаем небольшое движение через 2 секунды
        movement_start_time = 2.0

        while viewer.is_running():
            step_start = time.time()

            # Автоматическое включение движения через некоторое время
            current_time = time.time() - start_time
            if current_time > movement_start_time and controller.target_velocity == 0:
                controller.target_velocity = 0.2  # 0.2 м/с

            if not paused:
                # Применяем управление
                controller.apply_control()

                # Выполняем шаг симуляции
                mujoco.mj_step(m, d)

                step_count += 1

                # Выводим информацию каждые 500 шагов
                if step_count % 500 == 0:
                    elapsed = time.time() - start_time
                    print(f"Шагов: {step_count}, Время: {elapsed:.1f}с")

                    # Информация о состоянии
                    torso_pitch = controller.get_torso_pitch()
                    print(f"Текущий наклон: {np.degrees(torso_pitch):.1f}°")
                    print(f"Целевая скорость: {controller.target_velocity:.2f} м/с")

                # Синхронизация визуализации
                viewer.sync()

            # Регулировка скорости для реального времени
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()