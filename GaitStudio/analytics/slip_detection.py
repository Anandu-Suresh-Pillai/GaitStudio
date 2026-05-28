import numpy as np

class SlipDetector:
    def __init__(self):
        # Threshold above which we classify the foot as slipping
        self.slip_threshold = 0.05 # m/s (5 cm/s)

    def detect_slip(self, leg_name, foot_contact, joint_velocities, body_linear_velocity, 
                    body_angular_velocity, foot_pos_body, leg_jacobian=None, R_body=None):
        """
        Estimates foot slippage relative to the ground.
        
        foot_contact: bool, True if foot is in contact with the ground
        joint_velocities: 3D vector [q1_dot, q2_dot, q3_dot]
        body_linear_velocity: 3D vector of body center linear velocity in world frame
        body_angular_velocity: 3D vector of body angular velocity in world frame
        foot_pos_body: 3D position of foot relative to body center (body frame)
        leg_jacobian: 3x3 Jacobian matrix (body frame).
        R_body: 3x3 rotation matrix mapping vectors from body frame to world frame.
        """
        if not foot_contact:
            # No slip if foot is in the air (swing phase)
            return {'is_slipping': False, 'slip_velocity': 0.0, 'slip_ratio': 0.0}
            
        # 1. Compute foot velocity relative to body using Jacobian (body frame): v_rel = J * q_dot
        if leg_jacobian is not None:
            v_foot_rel_body = leg_jacobian @ joint_velocities
        else:
            v_foot_rel_body = np.zeros(3)
            
        # 2. Frame Alignment: Rotate body-frame vectors into world frame
        if R_body is not None:
            foot_pos_world = R_body @ foot_pos_body
            v_foot_rel_world = R_body @ v_foot_rel_body
        else:
            foot_pos_world = foot_pos_body
            v_foot_rel_world = v_foot_rel_body
            
        # 3. Compute theoretical foot velocity in world frame:
        # v_foot_world = v_body_world + w_body_world x r_foot_world + v_foot_rel_world
        w_cross_r_world = np.cross(body_angular_velocity, foot_pos_world)
        v_foot_theoretical_world = body_linear_velocity + w_cross_r_world + v_foot_rel_world
        
        # 4. Extract horizontal slip components (parallel to flat ground plane)
        slip_vel_vector = v_foot_theoretical_world[:2]
        slip_velocity = float(np.linalg.norm(slip_vel_vector))
        
        # 5. Calculate slip ratio
        body_speed = np.linalg.norm(body_linear_velocity[:2])
        if body_speed < 0.02:
            slip_ratio = 1.0 if slip_velocity > self.slip_threshold else 0.0
        else:
            slip_ratio = float(slip_velocity / (body_speed + slip_velocity))
            
        is_slipping = slip_velocity > self.slip_threshold
        
        return {
            'is_slipping': is_slipping,
            'slip_velocity': slip_velocity,
            'slip_ratio': np.clip(slip_ratio, 0.0, 1.0)
        }