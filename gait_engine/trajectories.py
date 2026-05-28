import numpy as np
from math import comb as math_comb

def bernstein_poly(i, n, t):
    """
    The Bernstein polynomial of n-th degree.
    """
    return math_comb(n, i) * (t**i) * ((1 - t)**(n - i))

def bezier_curve(control_points, num_points=100):
    """
    Given a set of control points, return the Bezier curve.
    control_points: shape (N, D) where N is number of control points, D is dimension
    """
    n = len(control_points) - 1
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, control_points.shape[1]))
    for i in range(n + 1):
        curve += np.outer(bernstein_poly(i, n, t), control_points[i])
    return curve

class FootTrajectoryPlanner:
    def __init__(self):
        pass

    def get_swing_trajectory(self, start_pos, end_pos, step_height, phase):
        """
        Computes the target foot position at a specific swing phase using Bezier curve interpolation.
        
        start_pos: 3D vector [x, y, z] at lift-off
        end_pos: 3D vector [x, y, z] at target touchdown
        step_height: maximum vertical clearance height (float)
        phase: normalized swing phase progress in [0, 1]
        
        returns: 3D position [x, y, z]
        """
        # Clamp phase
        phase = np.clip(phase, 0.0, 1.0)
        
        # 1. Horizontal interpolation (X and Y)
        # Hermite S-curve for smooth velocity profile (zero velocity at endpoints)
        s = 3 * phase**2 - 2 * phase**3
        x_val = start_pos[0] + (end_pos[0] - start_pos[0]) * s
        y_val = start_pos[1] + (end_pos[1] - start_pos[1]) * s
        
        # 2. Vertical interpolation (Z) using 5th-order Bezier
        # Control points ensure:
        #   - Zero velocity at start/end (first two and last two points match)
        #   - Peak height of ~1.5 * step_height above nominal
        z_ctrl = np.array([
            start_pos[2],                       # P0: start
            start_pos[2],                       # P1: start (zero velocity)
            start_pos[2] + step_height * 1.5,   # P2: peak region
            end_pos[2] + step_height * 1.5,     # P3: peak region
            end_pos[2],                         # P4: end (zero velocity)
            end_pos[2]                          # P5: end
        ])
        
        # Compute 5th-order Bezier value for Z
        z_val = 0.0
        n = 5
        for i in range(n + 1):
            b_poly = math_comb(n, i) * (phase**i) * ((1 - phase)**(n - i))
            z_val += b_poly * z_ctrl[i]
            
        return np.array([x_val, y_val, z_val])

    def get_stance_trajectory(self, start_pos, end_pos, phase):
        """
        Computes target foot position during stance phase (foot on ground moving backward relative to base).
        
        start_pos: 3D vector [x, y, z] at touchdown
        end_pos: 3D vector [x, y, z] at takeoff
        phase: normalized stance phase progress in [0, 1]
        
        returns: 3D position [x, y, z]
        """
        phase = np.clip(phase, 0.0, 1.0)
        
        # Linear interpolation for stance to maintain constant speed relative to the base
        pos = start_pos + (end_pos - start_pos) * phase
        return pos
