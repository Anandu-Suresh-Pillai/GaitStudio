# Hardware preset configurations
QUADRUPED_PRESETS = {
    "Unitree A1": {
        "base_mass": 9.0,               # kg (Torso only)
        "base_dim": [0.267, 0.194, 0.114], # [Length, Width, Height]
        "hip_l": 0.081,                 # m
        "thigh_l": 0.200,               # m
        "calf_l": 0.200,                # m
        "gear_ratio": 9.1,              # reduction ratio
        "k_t": 0.16,                    # Nm/A
        "r_coil": 0.18,                 # Ohms
        "kp_joint": 400.0               # Joint position gain (Kp)
    },
    "Boston Dynamics Spot (Scaled)": {
        "base_mass": 20.0,
        "base_dim": [0.60, 0.25, 0.15],
        "hip_l": 0.110,
        "thigh_l": 0.300,
        "calf_l": 0.300,
        "gear_ratio": 12.0,
        "k_t": 0.22,
        "r_coil": 0.25,
        "kp_joint": 800.0
    },
    "Unitree Go1": {
        "base_mass": 6.0,
        "base_dim": [0.25, 0.18, 0.10],
        "hip_l": 0.080,
        "thigh_l": 0.213,
        "calf_l": 0.213,
        "gear_ratio": 9.1,
        "k_t": 0.15,
        "r_coil": 0.16,
        "kp_joint": 350.0
    },
    "Custom (User-Defined)": {
        "base_mass": 10.0,
        "base_dim": [0.45, 0.20, 0.10],
        "hip_l": 0.060,
        "thigh_l": 0.220,
        "calf_l": 0.220,
        "gear_ratio": 10.0,
        "k_t": 0.18,
        "r_coil": 0.20,
        "kp_joint": 500.0
    }
}

import numpy as np
import time

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

from ik.analytical_ik import LegIK

class SimulationManager:
    def __init__(self, mode="mujoco", terrain_type="flat", config=None):
        self.mode = mode if (mode == "mujoco" and MUJOCO_AVAILABLE) else "standalone"
        self.terrain_type = "flat"
        self.dt = 1.0 / 240.0
        self.sim_time = 0.0
        
        # Default to Custom preset if none provided
        if config is None:
            config = QUADRUPED_PRESETS["Custom (User-Defined)"]
            
        self.apply_config(config)

    def apply_config(self, config):
        """
        Reconfigures the robot's physical layout and updates the analytical solvers.
        """
        self.base_dim = config["base_dim"]
        self.base_mass = config["base_mass"]
        self.hip_l = config["hip_l"]
        self.thigh_l = config["thigh_l"]
        self.calf_l = config["calf_l"]
        self.kp_joint = config["kp_joint"]
        
        # Calculate standard offsets based on base dimension
        self.hip_offsets = {
            'FL': np.array([self.base_dim[0]/2 - 0.04, self.base_dim[1]/2, 0.0]),
            'FR': np.array([self.base_dim[0]/2 - 0.04, -self.base_dim[1]/2, 0.0]),
            'RL': np.array([-self.base_dim[0]/2 + 0.04, self.base_dim[1]/2, 0.0]),
            'RR': np.array([-self.base_dim[0]/2 + 0.04, -self.base_dim[1]/2, 0.0])
        }
        
        # Rebuild IK solvers with new dimensions
        self.ik_solvers = {
            'FL': LegIK('FL', y_sign=1.0, hip_length=self.hip_l, thigh_length=self.thigh_l, calf_length=self.calf_l),
            'FR': LegIK('FR', y_sign=-1.0, hip_length=self.hip_l, thigh_length=self.thigh_l, calf_length=self.calf_l),
            'RL': LegIK('RL', y_sign=1.0, hip_length=self.hip_l, thigh_length=self.thigh_l, calf_length=self.calf_l),
            'RR': LegIK('RR', y_sign=-1.0, hip_length=self.hip_l, thigh_length=self.thigh_l, calf_length=self.calf_l)
        }
        
        # Re-initialize state vectors
        self.state = {
            'base_pos': np.array([0.0, 0.0, self.thigh_l + self.calf_l * 0.8]),
            'base_rpy': np.array([0.0, 0.0, 0.0]),
            'base_vel_lin': np.array([0.0, 0.0, 0.0]),
            'base_vel_ang': np.array([0.0, 0.0, 0.0]),
            'joint_q': np.zeros(12),
            'joint_dq': np.zeros(12),
            'joint_tau': np.zeros(12),
            'foot_contacts': {'FL': True, 'FR': True, 'RL': True, 'RR': True},
            'foot_forces': {'FL': 0.0, 'FR': 0.0, 'RL': 0.0, 'RR': 0.0},
            'foot_positions_hip': {k: np.array([0.0, 0.0, -(self.thigh_l + self.calf_l * 0.8)]) for k in ['FL', 'FR', 'RL', 'RR']},
            'foot_positions_world': {k: np.array([0.0, 0.0, 0.0]) for k in ['FL', 'FR', 'RL', 'RR']}
        }
        
        # Rebuild model structure
        self.init_simulation()

    # Replace the generate_mujoco_mjcf method inside sim_manager.py with this:

    def generate_mujoco_mjcf(self):
        """
        Dynamically generates a well-tuned MJCF quadruped robot model for stable MuJoCo simulation.
        """
        xml = f"""
        <mujoco model="gait_quadruped">
            <compiler angle="radian" coordinate="local"/>
            <option timestep="{self.dt}" gravity="0 0 -9.81" solver="Newton" iterations="50" tolerance="1e-10"/>
            
            <default>
                <!-- Increased damping and armature to prevent oscillations -->
                <joint type="hinge" limited="true" armature="0.04" damping="5.0" frictionloss="0.1"/>
                <geom friction="1.5 0.005 0.001" solref="0.005 1.0" solimp="0.9 0.95 0.001" condim="3"/>
                <position ctrlrange="-3.14 3.14" kp="500" forcerange="-150 150"/>
            </default>
            
            <asset>
                <texture type="2d" name="ground_tex" builtin="checker" rgb1="0.1 0.1 0.12" rgb2="0.15 0.15 0.18" width="100" height="100"/>
                <material name="ground_mat" texture="ground_tex" texrepeat="8 8"/>
                
                <material name="torso_mat" rgba="0.15 0.15 0.18 1"/>
                <material name="hip_mat" rgba="0.5 0.1 0.8 1"/>
                <material name="thigh_mat" rgba="0.3 0.3 0.35 1"/>
                <material name="calf_mat" rgba="0.12 0.12 0.15 1"/>
                <material name="foot_mat" rgba="0.2 0.9 0.1 1"/>
            </asset>
            
            <worldbody>
                <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>
                <geom name="ground" type="plane" size="10 10 0.1" material="ground_mat"/>
                
                <body name="base" pos="0 0 0.30">
                    <freejoint name="root"/>
                    <geom type="box" size="{self.base_dim[0]/2} {self.base_dim[1]/2} {self.base_dim[2]/2}" material="torso_mat" mass="9.0"/>
                    <site name="imu_site" pos="0 0 0"/>
        """
        
        # Add 4 legs
        legs = [
            ("FL", 1, 1),
            ("FR", 1, -1),
            ("RL", -1, 1),
            ("RR", -1, -1)
        ]
        
        for name, xs, ys in legs:
            hx = xs * (self.base_dim[0]/2 - 0.04)
            hy = ys * (self.base_dim[1]/2)
            y_offset = ys * self.hip_l
            
            # Leg masses are tuned to match Unitree A1 specifications
            xml += f"""
                    <body name="{name}_hip" pos="{hx:.3f} {hy:.3f} 0">
                        <joint name="{name}_hip_joint" axis="1 0 0" range="-0.8 0.8"/>
                        <geom type="cylinder" size="{0.02:.3f} {self.hip_l/2:.3f}" quat="0.7071 0.7071 0 0" material="hip_mat" mass="0.4"/>
                        
                        <body name="{name}_thigh" pos="0 {y_offset:.3f} 0">
                            <joint name="{name}_thigh_joint" axis="0 1 0" range="-1.2 2.5"/>
                            <geom type="cylinder" size="{0.018:.3f} {self.thigh_l/2:.3f}" pos="0 0 {-self.thigh_l/2:.3f}" material="thigh_mat" mass="0.2"/>
                            
                            <body name="{name}_calf" pos="0 0 {-self.thigh_l:.3f}">
                                <joint name="{name}_calf_joint" axis="0 1 0" range="-2.5 -0.2"/>
                                <geom type="cylinder" size="{0.014:.3f} {self.calf_l/2:.3f}" pos="0 0 {-self.calf_l/2:.3f}" material="calf_mat" mass="0.1"/>
                                
                                <body name="{name}_foot" pos="0 0 {-self.calf_l:.3f}">
                                    <geom name="{name}_foot_geom" type="sphere" size="{0.025:.3f}" material="foot_mat" mass="0.05" friction="1.8 0.005 0.001"/>
                                </body>
                            </body>
                        </body>
                    </body>
            """
            
        xml += """
                </body>
            </worldbody>
            
            <actuator>
        """
        
        # Position actuators for stiff, lag-free PD joint control
        for name, _, _ in legs:
            xml += f"""
                <position name="{name}_hip_actuator" joint="{name}_hip_joint" kp="400" forcerange="-100 100"/>
                <position name="{name}_thigh_actuator" joint="{name}_thigh_joint" kp="550" forcerange="-180 180"/>
                <position name="{name}_calf_actuator" joint="{name}_calf_joint" kp="550" forcerange="-180 180"/>
            """
            
        xml += """
            </actuator>
            
            <sensor>
                <!-- IMU sensors at base COM site -->
                <accelerometer name="imu_accel" site="imu_site"/>
                <gyro name="imu_gyro" site="imu_site"/>
                <framequat name="imu_quat" objtype="site" objname="imu_site"/>
                
                <!-- Joint position sensors -->
                <jointpos name="FL_hip_pos" joint="FL_hip_joint"/>
                <jointpos name="FL_thigh_pos" joint="FL_thigh_joint"/>
                <jointpos name="FL_calf_pos" joint="FL_calf_joint"/>
                <jointpos name="FR_hip_pos" joint="FR_hip_joint"/>
                <jointpos name="FR_thigh_pos" joint="FR_thigh_joint"/>
                <jointpos name="FR_calf_pos" joint="FR_calf_joint"/>
                <jointpos name="RL_hip_pos" joint="RL_hip_joint"/>
                <jointpos name="RL_thigh_pos" joint="RL_thigh_joint"/>
                <jointpos name="RL_calf_pos" joint="RL_calf_joint"/>
                <jointpos name="RR_hip_pos" joint="RR_hip_joint"/>
                <jointpos name="RR_thigh_pos" joint="RR_thigh_joint"/>
                <jointpos name="RR_calf_pos" joint="RR_calf_joint"/>
                
                <!-- Joint torque/force sensors -->
                <jointactuatorfrc name="FL_hip_torque" joint="FL_hip_joint"/>
                <jointactuatorfrc name="FL_thigh_torque" joint="FL_thigh_joint"/>
                <jointactuatorfrc name="FL_calf_torque" joint="FL_calf_joint"/>
                <jointactuatorfrc name="FR_hip_torque" joint="FR_hip_joint"/>
                <jointactuatorfrc name="FR_thigh_torque" joint="FR_thigh_joint"/>
                <jointactuatorfrc name="FR_calf_torque" joint="FR_calf_joint"/>
                <jointactuatorfrc name="RL_hip_torque" joint="RL_hip_joint"/>
                <jointactuatorfrc name="RL_thigh_torque" joint="RL_thigh_joint"/>
                <jointactuatorfrc name="RL_calf_torque" joint="RL_calf_joint"/>
                <jointactuatorfrc name="RR_hip_torque" joint="RR_hip_joint"/>
                <jointactuatorfrc name="RR_thigh_torque" joint="RR_thigh_joint"/>
                <jointactuatorfrc name="RR_calf_torque" joint="RR_calf_joint"/>
            </sensor>
        </mujoco>
        """
        return xml

    def init_simulation(self):
        """
        Initializes the selected simulation backend.
        """
        if self.mode == "mujoco":
            print("[+] Initializing MuJoCo Physics Backend...")
            mjcf_xml = self.generate_mujoco_mjcf()
            
            # Load model
            self.mj_model = mujoco.MjModel.from_xml_string(mjcf_xml)
            self.mj_data = mujoco.MjData(self.mj_model)
            
            # Calculate nominal IK angles for a stable standing pose
            # Target: feet slightly splayed outward, body at nominal height
            nominal_q = {}
            standing_height = 0.25
            for leg in ['FL', 'FR', 'RL', 'RR']:
                y_sign = 1.0 if 'L' in leg else -1.0
                # Target foot position relative to hip: slightly outward in Y, straight down in Z
                target = np.array([0.0, y_sign * 0.03, -standing_height])
                sol = self.ik_solvers[leg].solve_ik(target)
                if sol is not None:
                    nominal_q[leg] = sol
                else:
                    nominal_q[leg] = (0.0, 0.45, -1.35)
                    
            # Print nominal standing angles for debugging
            for leg in ['FL', 'FR', 'RL', 'RR']:
                q1, q2, q3 = nominal_q[leg]
                fk_pos = self.ik_solvers[leg].solve_fk(q1, q2, q3)
                print(f"  [{leg}] IK angles: ({q1:.3f}, {q2:.3f}, {q3:.3f}) rad -> FK check: ({fk_pos[0]:.3f}, {fk_pos[1]:.3f}, {fk_pos[2]:.3f})")
            
            # Set starting height: match the standing height plus a small margin for the base half-height
            # The base geom center is at qpos[2], and feet should touch ground at z=0
            # With body_height=0.25, the base center should be at z ≈ 0.25 + base_half_h ≈ 0.25
            # Start slightly above to let it settle gently
            self.mj_data.qpos[2] = standing_height + 0.02  # 0.27m, tiny drop
            
            # Set quaternion to identity (w=1, x=y=z=0)
            self.mj_data.qpos[3] = 1.0
            self.mj_data.qpos[4] = 0.0
            self.mj_data.qpos[5] = 0.0
            self.mj_data.qpos[6] = 0.0
            
            # Set joint positions and actuator targets to nominal standing angles
            for i, leg in enumerate(['FL', 'FR', 'RL', 'RR']):
                q1, q2, q3 = nominal_q[leg]
                self.mj_data.qpos[7 + i*3] = q1
                self.mj_data.qpos[8 + i*3] = q2
                self.mj_data.qpos[9 + i*3] = q3
                
                self.mj_data.ctrl[i*3] = q1
                self.mj_data.ctrl[i*3 + 1] = q2
                self.mj_data.ctrl[i*3 + 2] = q3
                
            # Zero all velocities
            self.mj_data.qvel[:] = 0.0
                
            # Settle the physics: run many steps to let the robot reach equilibrium
            print("[+] Settling physics...")
            for _ in range(500):
                mujoco.mj_step(self.mj_model, self.mj_data)
            
            # Print settled state
            pos = self.mj_data.qpos[0:3]
            print(f"  Settled base position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            
            # Launch passive visualizer
            try:
                def on_key(keycode):
                    self.pending_keys.append(keycode)
                self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data, key_callback=on_key)
                # Configure camera
                self.mj_viewer.cam.distance = 1.6
                self.mj_viewer.cam.elevation = -20
                self.mj_viewer.cam.azimuth = 135
                print("[+] MuJoCo 3D Viewer launched successfully!")
            except Exception as e:
                print(f"[-] Viewer launching failed: {e}. Running headless MuJoCo.")
                self.mj_viewer = None
        else:
            print("[+] Initializing Standalone Kinematic/Dynamic Fallback Engine...")
            self.sim_time = 0.0
            
            # Standalone state variables
            self.sa_base_pos = np.array([0.0, 0.0, 0.28])
            self.sa_base_rpy = np.array([0.0, 0.0, 0.0])
            self.sa_base_vel_lin = np.array([0.0, 0.0, 0.0])
            self.sa_base_vel_ang = np.array([0.0, 0.0, 0.0])
            
            # Rolling filters for body inertia
            self.sa_target_vel_lin = np.array([0.0, 0.0, 0.0])
            self.sa_target_vel_ang = np.array([0.0, 0.0, 0.0])

    def step_single(self, foot_targets, gait_generator):
        """
        Steps the simulation forward by exactly one physics timestep self.dt,
        applying foot targets via IK. This allows a 240Hz control loop.
        """
        self.sim_time += self.dt
        
        # 1. Compute joint targets for each leg using Analytical IK
        joint_targets = np.zeros(12)
        joint_names = ['FL', 'FR', 'RL', 'RR']
        
        for i, leg in enumerate(joint_names):
            target = foot_targets[leg]
            
            # Solve Analytical IK
            sol = self.ik_solvers[leg].solve_ik(target)
            if sol is not None:
                joint_targets[i*3 : i*3+3] = sol
            else:
                # Keep nominal pose if out of reach
                joint_targets[i*3 : i*3+3] = [0.0, 0.5, -1.2]
                
        # 2. Step the active engine by exactly 1 step
        if self.mode == "mujoco":
            # Apply target angles to actuator ctrl registers
            for idx in range(12):
                self.mj_data.ctrl[idx] = joint_targets[idx]
                
            # Run exactly ONE physics step
            mujoco.mj_step(self.mj_model, self.mj_data)
                
            # Read state back from MuJoCo
            self.state['base_pos'] = self.mj_data.qpos[0:3].copy()
            
            # Root orientation (quaternion -> RPY)
            quat = self.mj_data.qpos[3:7].copy()  # [w, x, y, z]
            w, x, y, z = quat
            roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
            pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
            yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
            self.state['base_rpy'] = np.array([roll, pitch, yaw])
            
            # Velocities
            self.state['base_vel_lin'] = self.mj_data.qvel[0:3].copy()
            self.state['base_vel_ang'] = self.mj_data.qvel[3:6].copy()
            
            # Joint states
            self.state['joint_q'] = self.mj_data.qpos[7:19].copy()
            self.state['joint_dq'] = self.mj_data.qvel[6:18].copy()
            self.state['joint_tau'] = self.mj_data.qfrc_actuator[6:18].copy()
            
            # Extract contact states and forces
            for leg in joint_names:
                self.state['foot_contacts'][leg] = False
                self.state['foot_forces'][leg] = 0.0
                
            # Read contacts from MuJoCo collision manifold
            for j in range(self.mj_data.ncon):
                contact = self.mj_data.contact[j]
                
                geom1_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                geom2_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                
                for leg in joint_names:
                    foot_geom = f"{leg}_foot_geom"
                    if geom1_name == foot_geom or geom2_name == foot_geom:
                        self.state['foot_contacts'][leg] = True
                        
                        c_force = np.zeros(6)
                        mujoco.mj_contactForce(self.mj_model, self.mj_data, j, c_force)
                        self.state['foot_forces'][leg] += max(0.1, abs(float(c_force[0])))
                        
            # Sync Viewer
            if self.mj_viewer is not None:
                self.mj_viewer.sync()
                
        else:
            # --- STANDALONE SIMULATOR ---
            # Run exactly ONE step of standalone integration
            alpha_lin = 0.08
            alpha_ang = 0.12
            
            gait = gait_generator.gaits[gait_generator.gait_name]
            
            cmd_vx = gait_generator.stride_x * gait_generator.frequency
            cmd_vy = gait_generator.stride_y * gait_generator.frequency
            cmd_wz = gait_generator.yaw_rate
            
            # Apply inertia filters
            self.sa_base_vel_lin[0] = (1 - alpha_lin) * self.sa_base_vel_lin[0] + alpha_lin * cmd_vx
            self.sa_base_vel_lin[1] = (1 - alpha_lin) * self.sa_base_vel_lin[1] + alpha_lin * cmd_vy
            self.sa_base_vel_lin[2] = 0.0
            self.sa_base_vel_ang[2] = (1 - alpha_ang) * self.sa_base_vel_ang[2] + alpha_ang * cmd_wz
            
            # Integrate position and orientation
            yaw = self.sa_base_rpy[2]
            c_y, s_y = np.cos(yaw), np.sin(yaw)
            v_world_x = self.sa_base_vel_lin[0] * c_y - self.sa_base_vel_lin[1] * s_y
            v_world_y = self.sa_base_vel_lin[0] * s_y + self.sa_base_vel_lin[1] * c_y
            
            self.sa_base_pos[0] += v_world_x * self.dt
            self.sa_base_pos[1] += v_world_y * self.dt
            self.sa_base_rpy[2] += self.sa_base_vel_ang[2] * self.dt
            
            # Dynamic sway
            gait_phase = gait_generator.master_phase
            
            self.sa_base_rpy[0] = 0.025 * np.sin(gait_phase * 2.0 * np.pi) * (1.0 if cmd_vx != 0 or cmd_vy != 0 else 0.0)
            self.sa_base_rpy[1] = 0.015 * np.cos(gait_phase * 2.0 * np.pi) * (1.0 if cmd_vx != 0 or cmd_vy != 0 else 0.0)
            
            # Ground height: always flat
            self.sa_base_pos[2] = gait_generator.body_height
            
            # Sync variables into self.state
            self.state['base_pos'] = self.sa_base_pos.copy()
            self.state['base_rpy'] = self.sa_base_rpy.copy()
            self.state['base_vel_lin'] = self.sa_base_vel_lin.copy()
            self.state['base_vel_ang'] = self.sa_base_vel_ang.copy()
            
            # Joint states track targets with minor delay
            alpha_joint = 0.25
            self.state['joint_q'] = (1.0 - alpha_joint) * self.state['joint_q'] + alpha_joint * joint_targets
            
            # Estimate joint velocities
            self.state['joint_dq'] = (joint_targets - self.state['joint_q']) / max(self.dt, 1e-6)
            
            # Estimate joint torques
            self.state['joint_tau'] = 40.0 * (joint_targets - self.state['joint_q'])
            
            # Contact states from gait planner phase timings
            for leg in joint_names:
                self.state['foot_contacts'][leg] = bool(gait_generator.foot_states[leg])
                
                if self.state['foot_contacts'][leg]:
                    num_stance = sum(gait_generator.foot_states.values())
                    num_stance = max(1, num_stance)
                    
                    base_mass = 9.0
                    gravity = 9.81
                    static_force = (base_mass * gravity) / num_stance
                    
                    phase = gait_generator.foot_phases[leg]
                    impact = 15.0 * np.exp(-15.0 * phase) if phase < 0.1 else 0.0
                    
                    self.state['foot_forces'][leg] = float(static_force + impact)
                else:
                    self.state['foot_forces'][leg] = 0.0

        # 3. Calculate foot Cartesian coordinate positions relative to base and world
        R_body = gait_generator.get_rotation_matrix(self.state['base_rpy'][0], self.state['base_rpy'][1], self.state['base_rpy'][2])
        
        for leg in joint_names:
            l_idx = joint_names.index(leg)
            q1, q2, q3 = self.state['joint_q'][l_idx*3 : l_idx*3+3]
            
            # Forward Kinematics relative to Hip joint
            foot_hip = self.ik_solvers[leg].solve_fk(q1, q2, q3)
            self.state['foot_positions_hip'][leg] = foot_hip
            
            # World coordinate position
            hip_off = self.hip_offsets[leg]
            foot_world = self.state['base_pos'] + R_body @ (hip_off + foot_hip)
            self.state['foot_positions_world'][leg] = foot_world

    def step(self, foot_targets, gait_generator):
        """
        Steps the simulation forward by self.dt, applying foot targets via IK.
        """
        self.sim_time += self.dt
        
        # 1. Compute joint targets for each leg using Analytical IK
        joint_targets = np.zeros(12)
        joint_names = ['FL', 'FR', 'RL', 'RR']
        
        for i, leg in enumerate(joint_names):
            target = foot_targets[leg]
            
            # Solve Analytical IK
            sol = self.ik_solvers[leg].solve_ik(target)
            if sol is not None:
                joint_targets[i*3 : i*3+3] = sol
            else:
                # Keep nominal pose if out of reach
                joint_targets[i*3 : i*3+3] = [0.0, 0.5, -1.2]
                
        # 2. Step the active engine
        if self.mode == "mujoco":
            # Apply target angles to actuator ctrl registers
            for idx in range(12):
                self.mj_data.ctrl[idx] = joint_targets[idx]
                
            # Run physics sub-steps (4 sub-steps per GUI frame at 1/240s each = ~60Hz)
            substeps = 4
            for _ in range(substeps):
                mujoco.mj_step(self.mj_model, self.mj_data)
                
            # Read state back from MuJoCo
            # Root position (x, y, z)
            self.state['base_pos'] = self.mj_data.qpos[0:3].copy()
            
            # Root orientation (quaternion -> RPY)
            quat = self.mj_data.qpos[3:7].copy()  # [w, x, y, z]
            w, x, y, z = quat
            roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
            pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
            yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
            self.state['base_rpy'] = np.array([roll, pitch, yaw])
            
            # Velocities
            self.state['base_vel_lin'] = self.mj_data.qvel[0:3].copy()
            self.state['base_vel_ang'] = self.mj_data.qvel[3:6].copy()
            
            # Joint states
            self.state['joint_q'] = self.mj_data.qpos[7:19].copy()
            self.state['joint_dq'] = self.mj_data.qvel[6:18].copy()
            
            # Joint torques (active motor forces)
            self.state['joint_tau'] = self.mj_data.qfrc_actuator[6:18].copy()
            
            # Extract contact states and forces
            for leg in joint_names:
                self.state['foot_contacts'][leg] = False
                self.state['foot_forces'][leg] = 0.0
                
            # Read contacts from MuJoCo collision manifold
            for j in range(self.mj_data.ncon):
                contact = self.mj_data.contact[j]
                
                geom1_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                geom2_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                
                for leg in joint_names:
                    foot_geom = f"{leg}_foot_geom"
                    if geom1_name == foot_geom or geom2_name == foot_geom:
                        self.state['foot_contacts'][leg] = True
                        
                        c_force = np.zeros(6)
                        mujoco.mj_contactForce(self.mj_model, self.mj_data, j, c_force)
                        self.state['foot_forces'][leg] += max(0.1, abs(float(c_force[0])))
                        
            # Sync Viewer
            if self.mj_viewer is not None:
                self.mj_viewer.sync()
                
        else:
            # --- STANDALONE SIMULATOR ---
            # Run 4 steps of standalone integration to match MuJoCo's 240Hz physics rate
            for _ in range(4):
                alpha_lin = 0.08
                alpha_ang = 0.12
                
                gait = gait_generator.gaits[gait_generator.gait_name]
                
                cmd_vx = gait_generator.stride_x * gait_generator.frequency
                cmd_vy = gait_generator.stride_y * gait_generator.frequency
                cmd_wz = gait_generator.yaw_rate
                
                # Apply inertia filters
                self.sa_base_vel_lin[0] = (1 - alpha_lin) * self.sa_base_vel_lin[0] + alpha_lin * cmd_vx
                self.sa_base_vel_lin[1] = (1 - alpha_lin) * self.sa_base_vel_lin[1] + alpha_lin * cmd_vy
                self.sa_base_vel_lin[2] = 0.0
                self.sa_base_vel_ang[2] = (1 - alpha_ang) * self.sa_base_vel_ang[2] + alpha_ang * cmd_wz
                
                # Integrate position and orientation
                yaw = self.sa_base_rpy[2]
                c_y, s_y = np.cos(yaw), np.sin(yaw)
                v_world_x = self.sa_base_vel_lin[0] * c_y - self.sa_base_vel_lin[1] * s_y
                v_world_y = self.sa_base_vel_lin[0] * s_y + self.sa_base_vel_lin[1] * c_y
                
                self.sa_base_pos[0] += v_world_x * self.dt
                self.sa_base_pos[1] += v_world_y * self.dt
                self.sa_base_rpy[2] += self.sa_base_vel_ang[2] * self.dt
                
                # Dynamic sway
                gait_phase = gait_generator.master_phase
                
                self.sa_base_rpy[0] = 0.025 * np.sin(gait_phase * 2.0 * np.pi) * (1.0 if cmd_vx != 0 or cmd_vy != 0 else 0.0)
                self.sa_base_rpy[1] = 0.015 * np.cos(gait_phase * 2.0 * np.pi) * (1.0 if cmd_vx != 0 or cmd_vy != 0 else 0.0)
                
                # Ground height: always flat
                self.sa_base_pos[2] = gait_generator.body_height
                
                # Sync variables into self.state
                self.state['base_pos'] = self.sa_base_pos.copy()
                self.state['base_rpy'] = self.sa_base_rpy.copy()
                self.state['base_vel_lin'] = self.sa_base_vel_lin.copy()
                self.state['base_vel_ang'] = self.sa_base_vel_ang.copy()
                
                # Joint states track targets with minor delay
                alpha_joint = 0.25
                self.state['joint_q'] = (1.0 - alpha_joint) * self.state['joint_q'] + alpha_joint * joint_targets
                
                # Estimate joint velocities
                self.state['joint_dq'] = (joint_targets - self.state['joint_q']) / max(self.dt, 1e-6)
                
                # Estimate joint torques
                self.state['joint_tau'] = 40.0 * (joint_targets - self.state['joint_q'])
                
                # Contact states from gait planner phase timings
                for leg in joint_names:
                    self.state['foot_contacts'][leg] = bool(gait_generator.foot_states[leg])
                    
                    if self.state['foot_contacts'][leg]:
                        num_stance = sum(gait_generator.foot_states.values())
                        num_stance = max(1, num_stance)
                        
                        base_mass = 9.0
                        gravity = 9.81
                        static_force = (base_mass * gravity) / num_stance
                        
                        phase = gait_generator.foot_phases[leg]
                        impact = 15.0 * np.exp(-15.0 * phase) if phase < 0.1 else 0.0
                        
                        self.state['foot_forces'][leg] = float(static_force + impact)
                    else:
                        self.state['foot_forces'][leg] = 0.0

        # 3. Calculate foot Cartesian coordinate positions relative to base and world
        R_body = gait_generator.get_rotation_matrix(self.state['base_rpy'][0], self.state['base_rpy'][1], self.state['base_rpy'][2])
        
        for leg in joint_names:
            l_idx = joint_names.index(leg)
            q1, q2, q3 = self.state['joint_q'][l_idx*3 : l_idx*3+3]
            
            # Forward Kinematics relative to Hip joint
            foot_hip = self.ik_solvers[leg].solve_fk(q1, q2, q3)
            self.state['foot_positions_hip'][leg] = foot_hip
            
            # World coordinate position
            hip_off = self.hip_offsets[leg]
            foot_world = self.state['base_pos'] + R_body @ (hip_off + foot_hip)
            self.state['foot_positions_world'][leg] = foot_world

    def close(self):
        """
        Cleans up simulation visualizers.
        """
        if self.mode == "mujoco" and self.mj_viewer is not None:
            self.mj_viewer.close()
            print("[+] MuJoCo Physics viewer closed.")
