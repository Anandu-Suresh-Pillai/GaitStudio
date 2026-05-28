import numpy as np

class TerrainPreset:
    def __init__(self, terrain_type="flat", difficulty=1.0):
        self.terrain_type = terrain_type
        self.difficulty = difficulty
        
        # Grid parameters for heightfields
        self.grid_width = 4.0          # meters
        self.grid_length = 4.0         # meters
        self.resolution = 0.05         # grid cell size (5cm)
        
        self.num_rows = int(self.grid_length / self.resolution)
        self.num_cols = int(self.grid_width / self.resolution)

    def get_height_at(self, x, y):
        """
        Analytical elevation function z = f(x, y) for custom physics and path planning.
        """
        if self.terrain_type == "flat":
            return 0.0
            
        elif self.terrain_type == "slope":
            # 10 degree incline along X-axis
            angle = np.radians(10.0) * self.difficulty
            # Offset center to start flat
            if x < 0.2:
                return 0.0
            return (x - 0.2) * np.tan(angle)
            
        elif self.terrain_type == "stairs":
            # Staircase along X-axis
            step_width = 0.25         # m
            step_height = 0.03 * self.difficulty # m
            if x < 0.3:
                return 0.0
            step_num = int((x - 0.3) / step_width)
            # Cap steps
            step_num = max(0, min(10, step_num))
            return step_num * step_height
            
        elif self.terrain_type == "rough":
            # Perlin-like mathematical rough terrain
            # Sine wave combination with high frequency noise
            h = 0.02 * np.sin(2.5 * x) * np.cos(2.5 * y)
            h += 0.008 * np.cos(8.0 * x) * np.sin(8.0 * y)
            # Add some high-frequency random noise
            # (using deterministic seed to avoid fluctuating heights at same x,y)
            seed_hash = int((abs(x * 100) + abs(y * 100)) % 1000)
            np.random.seed(seed_hash)
            h += np.random.uniform(-0.004, 0.004) * self.difficulty
            
            # Start flat around origin
            dist_to_center = np.sqrt(x**2 + y**2)
            gate = np.clip((dist_to_center - 0.3) / 0.4, 0.0, 1.0)
            return h * gate
            
        elif self.terrain_type == "stones":
            # Stepping stones: periodic circular pads rising from a floor of -0.05m
            # Floor
            base_floor = -0.05
            
            # Find closest pad center
            # Pads are placed on a 0.4m grid
            grid_spacing = 0.4
            grid_x = round(x / grid_spacing) * grid_spacing
            grid_y = round(y / grid_spacing) * grid_spacing
            
            dist_to_pad = np.sqrt((x - grid_x)**2 + (y - grid_y)**2)
            pad_radius = 0.12
            
            if dist_to_pad < pad_radius:
                # Deterministic height for each pad
                pad_seed = int((abs(grid_x * 10) + abs(grid_y * 10)) % 100)
                np.random.seed(pad_seed)
                pad_height = np.random.uniform(0.01, 0.04) * self.difficulty
                
                # Smooth filter to make them cylindrical pads
                return pad_height
            else:
                return base_floor
                
        return 0.0

    def generate_heightfield_data(self):
        """
        Generates a 2D float array containing heights for the simulator heightfield grid.
        Returns: 1D flat list (for Pybullet) and 2D grid
        """
        grid = np.zeros((self.num_rows, self.num_cols))
        
        # Center of the grid in world coordinates
        center_x = 0.0
        center_y = 0.0
        
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                # Map grid cell to world coordinate
                world_x = center_x + (r - self.num_rows / 2) * self.resolution
                world_y = center_y + (c - self.num_cols / 2) * self.resolution
                
                grid[r, c] = self.get_height_at(world_x, world_y)
                
        return grid.flatten().tolist(), grid

    def create_in_pybullet(self, pybullet_client):
        """
        Creates the physical terrain in PyBullet based on the preset.
        Returns: multibody ID of the terrain
        """
        p = pybullet_client
        
        if self.terrain_type == "flat":
            # Default flat ground plane
            terrain_id = p.loadURDF("plane.urdf")
            # Set friction
            p.changeDynamics(terrain_id, -1, lateralFriction=1.0)
            return terrain_id
            
        elif self.terrain_type in ["rough", "stones"]:
            # Create a PyBullet heightfield
            flat_heights, _ = self.generate_heightfield_data()
            
            # Create collision shape
            terrain_shape = p.createCollisionShape(
                shapeType=p.GEOM_HEIGHTFIELD,
                meshScale=[self.resolution, self.resolution, 1.0],
                heightfieldTextureScaling=(self.num_rows - 1) / 2,
                heightfieldData=flat_heights,
                numRows=self.num_rows,
                numColumns=self.num_cols
            )
            
            # Create visual shape (optional, uses visualizer)
            terrain_id = p.createMultiBody(
                baseMass=0, # static
                baseCollisionShapeIndex=terrain_shape,
                baseVisualShapeIndex=-1,
                basePosition=[0, 0, 0]
            )
            
            # Set high friction
            p.changeDynamics(terrain_id, -1, lateralFriction=1.2)
            
            # Renders as a dark brown grids
            p.changeVisualShape(terrain_id, -1, rgbaColor=[0.18, 0.16, 0.15, 1.0])
            return terrain_id
            
        elif self.terrain_type == "stairs":
            # Build stairs using multiple box shapes to make them look clean and crisp
            # rather than heightfields which have sloped mesh edges.
            step_width = 0.25
            step_height = 0.03 * self.difficulty
            
            # Standard plane for ground
            p.loadURDF("plane.urdf")
            
            # Steps
            stair_ids = []
            for i in range(1, 8):
                h = i * step_height
                # Box visual / collision
                box_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[step_width / 2, 1.5, h / 2])
                box_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[step_width / 2, 1.5, h / 2], rgbaColor=[0.25, 0.25, 0.28, 1.0])
                
                # Position step
                x_pos = 0.3 + (i - 1) * step_width + step_width / 2
                z_pos = h / 2
                
                step_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=box_shape,
                    baseVisualShapeIndex=box_visual,
                    basePosition=[x_pos, 0.0, z_pos]
                )
                p.changeDynamics(step_id, -1, lateralFriction=1.0)
                stair_ids.append(step_id)
                
            return stair_ids[0] # Return one of the IDs for tracking
            
        elif self.terrain_type == "slope":
            # Slanted box
            angle = np.radians(10.0) * self.difficulty
            p.loadURDF("plane.urdf")
            
            # Slope box
            slope_len = 3.0
            slope_thick = 0.1
            
            # Position slope
            x_pos = 0.2 + (slope_len / 2) * np.cos(angle)
            z_pos = (slope_len / 2) * np.sin(angle) - (slope_thick / 2) * np.cos(angle)
            
            box_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[slope_len / 2, 1.5, slope_thick / 2])
            box_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[slope_len / 2, 1.5, slope_thick / 2], rgbaColor=[0.28, 0.28, 0.32, 1.0])
            
            # Rotation quaternion for pitch angle
            q_slope = p.getQuaternionFromEuler([0.0, angle, 0.0])
            
            slope_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=box_shape,
                baseVisualShapeIndex=box_visual,
                basePosition=[x_pos, 0.0, z_pos],
                baseOrientation=q_slope
            )
            p.changeDynamics(slope_id, -1, lateralFriction=1.0)
            return slope_id
            
        return p.loadURDF("plane.urdf")
