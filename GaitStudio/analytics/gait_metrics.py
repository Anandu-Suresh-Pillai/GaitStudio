import numpy as np

class GaitMetricsTracker:
    def __init__(self, history_len=200):
        """
        Tracks rolling performance metrics of the quadruped gait.
        """
        self.history_len = history_len
        self.reset()

    def reset(self):
        # Rolling histories for active contact states (1=contact, 0=air)
        self.contact_history = {leg: [] for leg in ['FL', 'FR', 'RL', 'RR']}
        
        # Track current phase times
        self.current_state_durations = {leg: 0.0 for leg in ['FL', 'FR', 'RL', 'RR']}
        self.last_stance_durations = {leg: 0.0 for leg in ['FL', 'FR', 'RL', 'RR']}
        self.last_swing_durations = {leg: 0.0 for leg in ['FL', 'FR', 'RL', 'RR']}
        
        # Keep track of previous contact states to detect transitions
        self.prev_contacts = {leg: True for leg in ['FL', 'FR', 'RL', 'RR']}
        
        # Velocity tracking histories
        self.target_vels = []
        self.actual_vels = []
        
        # COM drift tracking
        # We integrate the lateral deviation of the COM from the starting line
        self.com_drift_y = 0.0

    def update(self, current_contacts, target_vel, actual_vel, dt):
        """
        Updates histories and calculates metrics.
        current_contacts: dict {'FL': bool, ...}
        target_vel: 3D vector [vx, vy, wz] (command)
        actual_vel: 3D vector [vx, vy, wz] (actual)
        """
        # 1. Update contact history and phase durations
        for leg in ['FL', 'FR', 'RL', 'RR']:
            contact = bool(current_contacts[leg])
            
            # Update rolling window
            self.contact_history[leg].append(1 if contact else 0)
            if len(self.contact_history[leg]) > self.history_len:
                self.contact_history[leg].pop(0)
                
            # Transition detection & duration integration
            if contact == self.prev_contacts[leg]:
                self.current_state_durations[leg] += dt
            else:
                # State changed!
                if self.prev_contacts[leg]: # Stance -> Swing
                    self.last_stance_durations[leg] = self.current_state_durations[leg]
                else: # Swing -> Stance
                    self.last_swing_durations[leg] = self.current_state_durations[leg]
                    
                # Reset clock
                self.current_state_durations[leg] = dt
                self.prev_contacts[leg] = contact
                
        # 2. Update velocities
        self.target_vels.append(np.array(target_vel))
        self.actual_vels.append(np.array(actual_vel))
        if len(self.target_vels) > self.history_len:
            self.target_vels.pop(0)
            self.actual_vels.pop(0)
            
        # 3. Integrate lateral drift
        # If we are supposed to go forward (vy = 0), then actual_vel[1] is drift velocity
        self.com_drift_y += actual_vel[1] * dt

    def get_metrics(self):
        """
        Computes the final metrics.
        returns: dict
        """
        metrics = {}
        
        # 1. Duty Factors
        duty_factors = {}
        for leg in ['FL', 'FR', 'RL', 'RR']:
            hist = self.contact_history[leg]
            if len(hist) > 0:
                duty_factors[leg] = sum(hist) / len(hist)
            else:
                duty_factors[leg] = 0.5
        metrics['duty_factors'] = duty_factors
        
        # Average stance/swing durations
        metrics['stance_durations'] = self.last_stance_durations.copy()
        metrics['swing_durations'] = self.last_swing_durations.copy()
        
        # 2. Gait Symmetry
        # A perfectly symmetric gait has diagonal legs with identical duty factors, 
        # and phase offsets that are exactly 0.5 cycles apart.
        # Let's measure symmetry as the difference between left and right duty factors.
        fl_fr_diff = abs(duty_factors['FL'] - duty_factors['FR'])
        rl_rr_diff = abs(duty_factors['RL'] - duty_factors['RR'])
        symmetry = 1.0 - np.clip((fl_fr_diff + rl_rr_diff) / 2.0, 0.0, 1.0)
        metrics['gait_symmetry'] = float(symmetry)
        
        # 3. Velocity Tracking Error (RMS)
        if len(self.target_vels) > 0:
            t_vels = np.array(self.target_vels)
            a_vels = np.array(self.actual_vels)
            err = np.linalg.norm(t_vels - a_vels, axis=1)
            metrics['velocity_tracking_error'] = float(np.mean(err))
        else:
            metrics['velocity_tracking_error'] = 0.0
            
        # 4. COM Drift
        metrics['com_drift_y'] = float(self.com_drift_y)
        
        # 5. Foot Contact Consistency
        # Measures if feet are touching down in a steady, rhythmic fashion
        # Calculate standard deviation of combined contact states
        total_contacts = []
        for i in range(len(self.contact_history['FL'])):
            cnt = sum(self.contact_history[leg][i] for leg in ['FL', 'FR', 'RL', 'RR'])
            total_contacts.append(cnt)
            
        if len(total_contacts) > 1:
            metrics['contact_consistency'] = float(1.0 - np.clip(np.std(total_contacts)/2.0, 0.0, 1.0))
        else:
            metrics['contact_consistency'] = 1.0
            
        return metrics
