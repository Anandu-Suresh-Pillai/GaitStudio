import numpy as np

class LegIK:
    def __init__(self, leg_name="FL", y_sign=1.0, hip_length=0.06, thigh_length=0.22, calf_length=0.22):
        """
        Analytical 3-DOF IK solver for a quadruped leg.
        leg_name: name of the leg (FL, FR, RL, RR)
        y_sign: +1.0 for Left legs (FL, RL), -1.0 for Right legs (FR, RR)
        hip_length: length of the hip abduction offset (L_hip = 0.06)
        thigh_length: thigh link length (0.22)
        calf_length: calf link length (0.22)
        
        Coordinate frame (hip joint frame, consistent with MJCF):
          X = forward
          Y = lateral (positive = left)
          Z = vertical (positive = up)
          
        Hip joint rotates around X-axis (abduction/adduction).
        Thigh joint rotates around Y-axis (pitch).
        Calf joint rotates around Y-axis (pitch).
        
        At zero joint angles, the leg hangs straight down along -Z,
        with the hip offset displacing the thigh joint along Y by y_sign * hip_length.
        """
        self.leg_name = leg_name
        self.y_sign = y_sign
        self.l1 = hip_length
        self.l2 = thigh_length
        self.l3 = calf_length
        
        # Joint limits (matching MJCF model)
        self.limits = {
            "hip": (-0.8, 0.8),
            "thigh": (-1.2, 2.5),
            "calf": (-2.5, -0.2)
        }

    def solve_ik(self, target_xyz):
        """
        Computes joint angles (q1, q2, q3) for a target foot position (x, y, z)
        relative to the hip joint origin.
        
        target_xyz: [x, y, z] target in hip joint frame
        returns: (q1, q2, q3) in radians, or None if unreachable
        """
        x, y, z = target_xyz
        
        # =====================================================
        # Step 1: Solve Hip Abduction (q1) — rotation around X
        # =====================================================
        # The hip link offsets the thigh joint along Y by y_sign * l1.
        # After hip rotation q1, the effective offset in the Y-Z plane is:
        #   y_hip = y_sign * l1 * cos(q1)
        #   z_hip = y_sign * l1 * sin(q1)
        # The thigh+calf assembly then extends in the pitch plane (X, Z' plane)
        # where Z' is the local vertical after hip rotation.
        #
        # In the Y-Z plane, the foot position (y, z) must satisfy:
        #   y = y_sign * l1 * cos(q1) + L_leg_yz * sin(q1 + alpha)  ... but the thigh plane
        #     contribution in Y-Z is only through the hip rotation.
        #
        # Simpler approach: project foot onto Y-Z plane.
        # The distance from hip origin to foot in Y-Z:
        d_yz = np.sqrt(y**2 + z**2)
        
        # The hip offset length
        l_hip = self.y_sign * self.l1
        
        # After the hip joint, the thigh-calf assembly hangs in the local Z' direction.
        # The foot's Y-Z position is:
        #   y = l_hip * cos(q1) - r * sin(q1)
        #   z = l_hip * sin(q1) + r * cos(q1)
        # where r is the projection of the thigh-calf onto the local -Z' axis (always negative for a hanging leg).
        # This means: [y, z] = Rot_x(q1) * [l_hip, r]
        # So: q1 = atan2(y, z) is NOT right.
        # Instead: q1 = atan2(z, y) - atan2(r, l_hip)
        # But we don't know r yet. However, r = sqrt(d_yz^2 - l_hip^2) with correct sign.
        
        # Guard against unreachable positions inside hip cylinder
        hip_sq = self.l1**2
        if d_yz**2 < hip_sq * 0.25:
            d_yz = self.l1 * 0.5
            
        r_sq = d_yz**2 - hip_sq
        if r_sq < 0:
            r_sq = 0.0
        r = -np.sqrt(r_sq)  # negative because the leg hangs DOWN
        
        # q1 = atan2(z, y) - atan2(r, l_hip)
        q1 = np.arctan2(z, y) - np.arctan2(r, l_hip)
        
        # =====================================================
        # Step 2: Transform target into the thigh pitch plane
        # =====================================================
        # Rotate the target point by -q1 around the X axis to get into the hip-local frame
        # In the rotated frame, y' should equal l_hip and z' is the reach in the pitch plane.
        cq1 = np.cos(q1)
        sq1 = np.sin(q1)
        
        # After rotating by -q1:
        # y' = y * cos(q1) + z * sin(q1)   -> should be ~l_hip
        # z' = -y * sin(q1) + z * cos(q1)  -> pitch-plane reach (negative = downward)
        x_p = x
        z_p = -y * sq1 + z * cq1
        
        # =====================================================
        # Step 3: Solve 2-link planar IK for thigh and calf (q2, q3)
        # =====================================================
        # In the pitch plane, the foot is at (x_p, z_p) from the thigh joint.
        # The thigh link has length l2, calf link has length l3.
        # At q2=0, q3=0: foot is at (0, -(l2+l3)) = straight down.
        
        d_sq = x_p**2 + z_p**2
        d_leg = np.sqrt(d_sq)
        
        # Clamp reach to feasible range
        max_reach = self.l2 + self.l3 - 0.001
        min_reach = abs(self.l2 - self.l3) + 0.001
        if d_leg > max_reach:
            # Scale target to max reach
            scale = max_reach / d_leg
            x_p *= scale
            z_p *= scale
            d_sq = x_p**2 + z_p**2
            d_leg = max_reach
        elif d_leg < min_reach:
            d_leg = min_reach
            d_sq = d_leg**2
        
        # Law of cosines for q3 (calf angle)
        # d^2 = l2^2 + l3^2 + 2*l2*l3*cos(q3)
        # Note: when q3=0, cos(q3)=1, d=l2+l3 (fully extended)
        c3 = (d_sq - self.l2**2 - self.l3**2) / (2.0 * self.l2 * self.l3)
        c3 = np.clip(c3, -1.0, 1.0)
        
        # q3 is negative for a backward-bending knee (per MJCF joint limits)
        q3 = -np.arccos(c3)
        
        # Solve q2 (thigh angle)
        # The foot position in the pitch plane:
        #   x_p = l2 * sin(q2) + l3 * sin(q2 + q3)
        #   z_p = -l2 * cos(q2) - l3 * cos(q2 + q3)
        # Using auxiliary variables:
        k1 = self.l2 + self.l3 * c3
        k2 = self.l3 * np.sin(q3)  # sin(q3) is negative
        
        # q2 = atan2(x_p, -z_p) - atan2(k2, k1)
        q2 = np.arctan2(x_p, -z_p) - np.arctan2(k2, k1)
        
        # =====================================================
        # Step 4: Clamp to joint limits
        # =====================================================
        q1 = np.clip(q1, self.limits["hip"][0], self.limits["hip"][1])
        q2 = np.clip(q2, self.limits["thigh"][0], self.limits["thigh"][1])
        q3 = np.clip(q3, self.limits["calf"][0], self.limits["calf"][1])
        
        return float(q1), float(q2), float(q3)

    def solve_fk(self, q1, q2, q3):
        """
        Computes the Cartesian foot position (x, y, z) in the hip joint frame
        given joint angles (q1, q2, q3).
        
        At q1=q2=q3=0, the foot is at (0, y_sign*l1, -(l2+l3)).
        """
        # Pitch plane kinematics (thigh + calf)
        x_p = self.l2 * np.sin(q2) + self.l3 * np.sin(q2 + q3)
        z_p = -self.l2 * np.cos(q2) - self.l3 * np.cos(q2 + q3)
        y_p = self.y_sign * self.l1
        
        # Rotate back around X-axis by q1
        x = x_p
        y = y_p * np.cos(q1) - z_p * np.sin(q1)
        z = y_p * np.sin(q1) + z_p * np.cos(q1)
        
        return np.array([x, y, z])
