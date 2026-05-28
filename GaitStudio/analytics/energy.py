import numpy as np

class EnergyEstimator:
    def __init__(self, num_joints=12):
        """
        Estimates power and energy consumption of the quadruped robot.
        """
        self.num_joints = num_joints
        
        # Motor parameters (approximate Spot/A1 BLDC actuator)
        self.r_coil = 0.2              # Ohms (coil resistance)
        self.k_t = 0.18               # Nm/A (motor torque constant)
        self.gear_ratio = 9.0         # 9:1 gear reduction (converts joint torque to motor torque)
        self.p_standby = 8.0          # W (standby power of electronics/onboard computer)
        
        # Cumulative energy metrics
        self.total_energy_joules = 0.0
        self.total_mech_work_joules = 0.0

    def reset(self):
        self.total_energy_joules = 0.0
        self.total_mech_work_joules = 0.0

    def compute_power(self, joint_torques, joint_velocities, regen_efficiency=0.0):
        """
        joint_torques: array or list of 12 joint torques (Nm)
        joint_velocities: array or list of 12 joint angular velocities (rad/s)
        regen_efficiency: efficiency of regenerative braking in [0.0, 1.0]
        """
        joint_torques = np.array(joint_torques)
        joint_velocities = np.array(joint_velocities)
        
        # 1. Mechanical Power: P_mech = sum( tau * omega )
        mech_powers = joint_torques * joint_velocities
        p_mech_pos = float(np.sum(np.maximum(0.0, mech_powers)))
        p_mech_neg = float(np.sum(np.minimum(0.0, mech_powers)))
        
        # 2. Actuator Copper Losses (I^2 * R) using the gear reduction ratio
        # Motor Torque = Joint Torque / Gear Ratio
        motor_torques = joint_torques / self.gear_ratio
        currents = motor_torques / self.k_t
        copper_losses = self.r_coil * (currents**2)
        p_loss = float(np.sum(copper_losses))
        
        # 3. Total Electrical Power:
        # If mechanical power is negative (braking), recover it with regen_efficiency
        p_mech_net = p_mech_pos + (regen_efficiency * p_mech_neg)
        p_elec = max(self.p_standby, p_mech_net + p_loss + self.p_standby)
        
        return {
            'mechanical_power': float(np.sum(np.abs(mech_powers))),
            'electrical_power': p_elec,
            'motor_losses': p_loss
        }

    def update_energy(self, p_elec, p_mech, dt):
        """
        Integrates power over time to calculate cumulative energy consumption (Joules).
        """
        self.total_energy_joules += p_elec * dt
        self.total_mech_work_joules += p_mech * dt
        
        energy_wh = self.total_energy_joules / 3600.0
        mech_work_wh = self.total_mech_work_joules / 3600.0
        
        return {
            'total_energy_wh': energy_wh,
            'total_mech_work_wh': mech_work_wh
        }