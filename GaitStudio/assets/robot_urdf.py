import os

def generate_urdf(output_path="assets/robot.urdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Robot parameters
    base_l, base_w, base_h = 0.45, 0.20, 0.10
    base_mass = 9.0
    
    hip_l, hip_r = 0.06, 0.02
    hip_mass = 0.6
    
    thigh_l, thigh_r = 0.22, 0.018
    thigh_mass = 1.0
    
    calf_l, calf_r = 0.22, 0.014
    calf_mass = 0.4
    
    foot_r = 0.02
    foot_mass = 0.1
    
    # Colors
    color_torso = "0.15 0.15 0.18 1"       # Deep Slate
    color_hip = "0.5 0.1 0.8 1"           # Purple/Blue accent
    color_thigh = "0.2 0.2 0.25 1"         # Slate gray
    color_calf = "0.1 0.1 0.12 1"          # Dark gray
    color_foot = "0.2 0.9 0.1 1"           # Neon Green
    
    urdf = []
    urdf.append('<?xml version="1.0"?>')
    urdf.append('<robot name="gait_quadruped">')
    
    # Materials definitions
    urdf.append(f'  <material name="torso_mat"><color rgba="{color_torso}"/></material>')
    urdf.append(f'  <material name="hip_mat"><color rgba="{color_hip}"/></material>')
    urdf.append(f'  <material name="thigh_mat"><color rgba="{color_thigh}"/></material>')
    urdf.append(f'  <material name="calf_mat"><color rgba="{color_calf}"/></material>')
    urdf.append(f'  <material name="foot_mat"><color rgba="{color_foot}"/></material>')
    
    # Base link (torso)
    urdf.append('  <link name="base_link">')
    urdf.append('    <inertial>')
    urdf.append(f'      <mass value="{base_mass}"/>')
    urdf.append('      <origin xyz="0 0 0"/>')
    urdf.append(f'      <inertia ixx="{base_mass * (base_w**2 + base_h**2) / 12:.6f}" ixy="0" ixz="0" ')
    urdf.append(f'               iyy="{base_mass * (base_l**2 + base_h**2) / 12:.6f}" iyz="0" ')
    urdf.append(f'               izz="{base_mass * (base_l**2 + base_w**2) / 12:.6f}"/>')
    urdf.append('    </inertial>')
    urdf.append('    <visual>')
    urdf.append('      <origin xyz="0 0 0"/>')
    urdf.append('      <geometry>')
    urdf.append(f'        <box size="{base_l} {base_w} {base_h}"/>')
    urdf.append('      </geometry>')
    urdf.append('      <material name="torso_mat"/>')
    urdf.append('    </visual>')
    urdf.append('    <collision>')
    urdf.append('      <origin xyz="0 0 0"/>')
    urdf.append('      <geometry>')
    urdf.append(f'        <box size="{base_l} {base_w} {base_h}"/>')
    urdf.append('      </geometry>')
    urdf.append('    </collision>')
    urdf.append('  </link>')
    
    # Define the 4 legs
    # Leg placements relative to base center (x_sign, y_sign)
    legs = [
        ("FL", 1, 1),
        ("FR", 1, -1),
        ("RL", -1, 1),
        ("RR", -1, -1)
    ]
    
    for name, xs, ys in legs:
        # Offsets
        hx_off = xs * (base_l / 2 - 0.04)
        hy_off = ys * (base_w / 2 + hip_l / 2)
        hz_off = 0.0
        
        # 1. Hip Joint (abduction/adduction)
        urdf.append(f'  <joint name="{name}_hip_joint" type="revolute">')
        urdf.append(f'    <parent link="base_link"/>')
        urdf.append(f'    <child link="{name}_hip_link"/>')
        urdf.append(f'    <origin xyz="{hx_off:.3f} {hy_off:.3f} {hz_off:.3f}" rpy="0 0 0"/>')
        urdf.append(f'    <axis xyz="1 0 0"/>') # Roll
        urdf.append(f'    <limit effort="40.0" velocity="20.0" lower="-0.8" upper="0.8"/>')
        urdf.append('  </joint>')
        
        # Hip Link
        urdf.append(f'  <link name="{name}_hip_link">')
        urdf.append('    <inertial>')
        urdf.append(f'      <mass value="{hip_mass}"/>')
        # Cylinder aligned with Y-axis for visualization
        urdf.append(f'      <origin xyz="0 {ys * hip_l/4:.3f} 0"/>')
        urdf.append(f'      <inertia ixx="0.0003" ixy="0" ixz="0" iyy="0.0004" iyz="0" izz="0.0003"/>')
        urdf.append('    </inertial>')
        urdf.append('    <visual>')
        urdf.append(f'      <origin xyz="0 0 0" rpy="1.5708 0 0"/>')
        urdf.append('      <geometry>')
        urdf.append(f'        <cylinder radius="{hip_r:.3f}" length="{hip_l:.3f}"/>')
        urdf.append('      </geometry>')
        urdf.append('      <material name="hip_mat"/>')
        urdf.append('    </visual>')
        urdf.append('    <collision>')
        urdf.append(f'      <origin xyz="0 0 0" rpy="1.5708 0 0"/>')
        urdf.append('      <geometry>')
        urdf.append(f'        <cylinder radius="{hip_r:.3f}" length="{hip_l:.3f}"/>')
        urdf.append('      </geometry>')
        urdf.append('    </collision>')
        urdf.append('  </link>')
        
        # 2. Thigh Joint (pitch)
        tx_off = 0.0
        ty_off = ys * hip_l
        tz_off = 0.0
        urdf.append(f'  <joint name="{name}_thigh_joint" type="revolute">')
        urdf.append(f'    <parent link="{name}_hip_link"/>')
        urdf.append(f'    <child link="{name}_thigh_link"/>')
        urdf.append(f'    <origin xyz="{tx_off:.3f} {ty_off:.3f} {tz_off:.3f}" rpy="0 0 0"/>')
        urdf.append(f'    <axis xyz="0 1 0"/>') # Pitch
        urdf.append(f'    <limit effort="60.0" velocity="15.0" lower="-1.2" upper="2.5"/>')
        urdf.append('  </joint>')
        
        # Thigh Link
        urdf.append(f'  <link name="{name}_thigh_link">')
        urdf.append('    <inertial>')
        urdf.append(f'      <mass value="{thigh_mass}"/>')
        urdf.append(f'      <origin xyz="0 0 {-thigh_l/2:.3f}"/>')
        urdf.append(f'      <inertia ixx="{thigh_mass * thigh_l**2 / 12:.6f}" ixy="0" ixz="0" ')
        urdf.append(f'               iyy="{thigh_mass * thigh_l**2 / 12:.6f}" iyz="0" ')
        urdf.append(f'               izz="{thigh_mass * thigh_r**2 / 2:.6f}"/>')
        urdf.append('    </inertial>')
        urdf.append('    <visual>')
        urdf.append(f'      <origin xyz="0 0 {-thigh_l/2:.3f}" rpy="0 0 0"/>')
        urdf.append('      <geometry>')
        urdf.append(f'        <cylinder radius="{thigh_r:.3f}" length="{thigh_l:.3f}"/>')
        urdf.append('      </geometry>')
        urdf.append('      <material name="thigh_mat"/>')
        urdf.append('    </visual>')
        urdf.append('    <collision>')
        urdf.append(f'      <origin xyz="0 0 {-thigh_l/2:.3f}" rpy="0 0 0"/>')
        urdf.append('      <geometry>')
        urdf.append(f'        <cylinder radius="{thigh_r:.3f}" length="{thigh_l:.3f}"/>')
        urdf.append('      </geometry>')
        urdf.append('    </collision>')
        urdf.append('  </link>')
        
        # 3. Calf Joint (pitch)
        cx_off = 0.0
        cy_off = 0.0
        cz_off = -thigh_l
        urdf.append(f'  <joint name="{name}_calf_joint" type="revolute">')
        urdf.append(f'    <parent link="{name}_thigh_link"/>')
        urdf.append(f'    <child link="{name}_calf_link"/>')
        urdf.append(f'    <origin xyz="{cx_off:.3f} {cy_off:.3f} {cz_off:.3f}" rpy="0 0 0"/>')
        urdf.append(f'    <axis xyz="0 1 0"/>') # Pitch
        urdf.append(f'    <limit effort="60.0" velocity="20.0" lower="-2.5" upper="-0.2"/>')
        urdf.append('  </joint>')
        
        # Calf Link
        urdf.append(f'  <link name="{name}_calf_link">')
        urdf.append('    <inertial>')
        urdf.append(f'      <mass value="{calf_mass}"/>')
        urdf.append(f'      <origin xyz="0 0 {-calf_l/2:.3f}"/>')
        urdf.append(f'      <inertia ixx="{calf_mass * calf_l**2 / 12:.6f}" ixy="0" ixz="0" ')
        urdf.append(f'               iyy="{calf_mass * calf_l**2 / 12:.6f}" iyz="0" ')
        urdf.append(f'               izz="{calf_mass * calf_r**2 / 2:.6f}"/>')
        urdf.append('    </inertial>')
        urdf.append('    <visual>')
        urdf.append(f'      <origin xyz="0 0 {-calf_l/2:.3f}" rpy="0 0 0"/>')
        urdf.append('      <geometry>')
        urdf.append(f'        <cylinder radius="{calf_r:.3f}" length="{calf_l:.3f}"/>')
        urdf.append('      </geometry>')
        urdf.append('      <material name="calf_mat"/>')
        urdf.append('    </visual>')
        urdf.append('    <collision>')
        urdf.append(f'      <origin xyz="0 0 {-calf_l/2:.3f}" rpy="0 0 0"/>')
        urdf.append('      <geometry>')
        urdf.append(f'        <cylinder radius="{calf_r:.3f}" length="{calf_l:.3f}"/>')
        urdf.append('      </geometry>')
        urdf.append('    </collision>')
        urdf.append('  </link>')
        
        # 4. Foot (fixed sphere)
        fx_off = 0.0
        fy_off = 0.0
        fz_off = -calf_l
        urdf.append(f'  <joint name="{name}_foot_joint" type="fixed">')
        urdf.append(f'    <parent link="{name}_calf_link"/>')
        urdf.append(f'    <child link="{name}_foot"/>')
        urdf.append(f'    <origin xyz="{fx_off:.3f} {fy_off:.3f} {fz_off:.3f}" rpy="0 0 0"/>')
        urdf.append('  </joint>')
        
        # Foot Link
        urdf.append(f'  <link name="{name}_foot">')
        urdf.append('    <inertial>')
        urdf.append(f'      <mass value="{foot_mass}"/>')
        urdf.append('      <origin xyz="0 0 0"/>')
        urdf.append(f'      <inertia ixx="0.00001" ixy="0" ixz="0" iyy="0.00001" iyz="0" izz="0.00001"/>')
        urdf.append('    </inertial>')
        urdf.append('    <visual>')
        urdf.append('      <origin xyz="0 0 0"/>')
        urdf.append('      <geometry>')
        urdf.append(f'        <sphere radius="{foot_r:.3f}"/>')
        urdf.append('      </geometry>')
        urdf.append('      <material name="foot_mat"/>')
        urdf.append('    </visual>')
        urdf.append('    <collision>')
        urdf.append('      <origin xyz="0 0 0"/>')
        urdf.append('      <geometry>')
        urdf.append(f'        <sphere radius="{foot_r:.3f}"/>')
        urdf.append('      </geometry>')
        # High friction on the foot
        urdf.append('      <contact>')
        urdf.append('        <lateral_friction value="1.2"/>')
        urdf.append('        <spinning_friction value="0.02"/>')
        urdf.append('        <rolling_friction value="0.02"/>')
        urdf.append('      </contact>')
        urdf.append('    </collision>')
        urdf.append('  </link>')
        
    urdf.append('</robot>')
    
    with open(output_path, 'w') as f:
        f.write("\n".join(urdf))
    print(f"Successfully generated URDF at {output_path}")

if __name__ == "__main__":
    generate_urdf()
