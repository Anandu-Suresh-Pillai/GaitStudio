import numpy as np
from gait_engine.trajectories import FootTrajectoryPlanner

class GaitGenerator:
    def __init__(self, hip_positions=None):
        """
        Coordinates phases and plans foot trajectories for a quadruped robot.
        hip_positions: dict of 3D vectors containing hip offsets from base center
                       {'FL': [x, y, z], ...}
        """
        # Nominal hip offsets from center of mass (default values if not provided)
        if hip_positions is None:
            self.hip_positions = {
                'FL': np.array([0.185, 0.10, 0.0]),
                'FR': np.array([0.185, -0.10, 0.0]),
                'RL': np.array([-0.185, 0.10, 0.0]),
                'RR': np.array([-0.185, -0.10, 0.0])
            }
        else:
            self.hip_positions = hip_positions
            
        self.planner = FootTrajectoryPlanner()
        
        # Core gait parameters (live adjustable)
        self.frequency = 1.5           # Hz (gait cycle frequency)
        self.stride_x = 0.0            # m (forward stride length)
        self.stride_y = 0.0            # m (lateral stride length)
        self.step_height = 0.06        # m (foot lift height)
        self.body_height = 0.25        # m (nominal base height from feet)
        self.yaw_rate = 0.0            # rad/s (yaw turning rate)
        
        # Base attitude offsets (live adjustable)
        self.roll_offset = 0.0         # rad
        self.pitch_offset = 0.0        # rad
        self.yaw_offset = 0.0          # rad
        
        # Master phase clock in [0, 1)
        self.master_phase = 0.0
        
        # Active gait name
        self.gait_name = 'trot'
        
        # Gait definitions: (duty_factor, phase_offsets)
        self.gaits = {
            'walk': {
                'duty_factor': 0.75,
                'offsets': {'FL': 0.0, 'FR': 0.5, 'RL': 0.75, 'RR': 0.25}
            },
            'trot': {
                'duty_factor': 0.50,
                'offsets': {'FL': 0.0, 'FR': 0.5, 'RL': 0.5, 'RR': 0.0}
            },
            'pace': {
                'duty_factor': 0.50,
                'offsets': {'FL': 0.0, 'FR': 0.5, 'RL': 0.0, 'RR': 0.5}
            },
            'bound': {
                'duty_factor': 0.50,
                'offsets': {'FL': 0.0, 'FR': 0.0, 'RL': 0.5, 'RR': 0.5}
            }
        }
        
        # Tracking foot states for logging and graphics
        self.foot_phases = {'FL': 0.0, 'FR': 0.0, 'RL': 0.0, 'RR': 0.0}
        self.foot_states = {'FL': 1, 'FR': 1, 'RL': 1, 'RR': 1} # 1 = Stance, 0 = Swing
        
        # To avoid discontinuities at touchdown/liftoff, we track initial/final foot positions
        self.last_swing_start = {k: np.array([0., 0., -self.body_height]) for k in ['FL', 'FR', 'RL', 'RR']}

    def reset(self):
        """
        Resets the gait generator's internal clock phase and swing landing coordinates.
        """
        self.master_phase = 0.0
        self.foot_phases = {'FL': 0.0, 'FR': 0.0, 'RL': 0.0, 'RR': 0.0}
        self.foot_states = {'FL': 1, 'FR': 1, 'RL': 1, 'RR': 1}
        self.last_swing_start = {k: np.array([0., 0., -self.body_height]) for k in ['FL', 'FR', 'RL', 'RR']}

    def set_gait(self, gait_name):
        if gait_name in self.gaits:
            self.gait_name = gait_name
            
    def update_phase(self, dt):
        """
        Increments the master clock based on time delta and frequency.
        """
        self.master_phase = (self.master_phase + self.frequency * dt) % 1.0
        return self.master_phase

    def get_rotation_matrix(self, roll, pitch, yaw):
        """
        Computes 3D rotation matrix.
        """
        c_r, s_r = np.cos(roll), np.sin(roll)
        c_p, s_p = np.cos(pitch), np.sin(pitch)
        c_y, s_y = np.cos(yaw), np.sin(yaw)
        
        R_x = np.array([[1, 0, 0], [0, c_r, -s_r], [0, s_r, c_r]])
        R_y = np.array([[c_p, 0, s_p], [0, 1, 0], [-s_p, 0, c_p]])
        R_z = np.array([[c_y, -s_y, 0], [s_y, c_y, 0], [0, 0, 1]])
        
        return R_z @ R_y @ R_x

    def compute_foot_targets(self):
        """
        Computes Cartesian target positions [x, y, z] for each foot relative to its hip joint.
        This includes linear and rotational locomotion commands + body attitude transforms.
        
        returns: dict of 3D foot vectors {'FL': [x, y, z], ...}
        """
        gait = self.gaits[self.gait_name]
        duty_factor = gait['duty_factor']
        offsets = gait['offsets']
        
        targets = {}
        
        # Compute body rotation matrix
        R_body = self.get_rotation_matrix(self.roll_offset, self.pitch_offset, self.yaw_offset)
        
        for leg in ['FL', 'FR', 'RL', 'RR']:
            # 1. Local leg phase
            leg_phase = (self.master_phase + offsets[leg]) % 1.0
            self.foot_phases[leg] = leg_phase
            
            # Nominal stance width and offset
            # Add outward offset in Y to prevent legs from colliding and improve stability
            y_sign = 1.0 if 'L' in leg else -1.0
            stance_width_offset = 0.03
            
            nominal_foot_hip = np.array([0.0, y_sign * stance_width_offset, -self.body_height])
            
            # 2. Compute Stride stroke vector (displacement during stance)
            # Yaw tanget: T = [-y_hip, x_hip]
            hip_pos = self.hip_positions[leg]
            yaw_tangent = np.array([-hip_pos[1], hip_pos[0], 0.0])
            
            # Stride vector
            stride_vec = np.array([self.stride_x, self.stride_y, 0.0]) + self.yaw_rate * yaw_tangent
            
            # 3. Determine Stance vs Swing
            if leg_phase < duty_factor:
                # --- STANCE PHASE ---
                self.foot_states[leg] = 1 # Stance
                
                # Normalized stance progress from 0 to 1
                phase_s = leg_phase / duty_factor
                
                # Foot travels from +stride_vec / 2 to -stride_vec / 2 relative to nominal footprint
                start_pos = nominal_foot_hip + stride_vec / 2
                end_pos = nominal_foot_hip - stride_vec / 2
                
                foot_pos_hip = self.planner.get_stance_trajectory(start_pos, end_pos, phase_s)
                
                # Record this position as the starting point for next swing
                self.last_swing_start[leg] = foot_pos_hip.copy()
            else:
                # --- SWING PHASE ---
                self.foot_states[leg] = 0 # Swing
                
                # Normalized swing progress from 0 to 1
                phase_w = (leg_phase - duty_factor) / (1.0 - duty_factor)
                
                # Starts at the last stance exit point, lands at +stride_vec / 2
                start_pos = self.last_swing_start[leg]
                end_pos = nominal_foot_hip + stride_vec / 2
                
                foot_pos_hip = self.planner.get_swing_trajectory(start_pos, end_pos, self.step_height, phase_w)
            
            # 4. Apply Base Attitude offsets (Roll, Pitch, Yaw)
            # target_in_body_frame = R_body.T @ (hip_pos + foot_pos_hip) - hip_pos
            # This rotates the target foot position relative to the hip in the opposite direction of body rotation
            foot_pos_body = R_body.T @ (hip_pos + foot_pos_hip) - hip_pos
            
            targets[leg] = foot_pos_body
            
        return targets
