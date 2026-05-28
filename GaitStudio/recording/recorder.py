import csv
import json
import os
import time

class LocomotionRecorder:
    def __init__(self, default_dir="recording/sessions"):
        self.default_dir = default_dir
        os.makedirs(self.default_dir, exist_ok=True)
        
        self.is_recording = False
        self.session_data = []
        self.start_time = 0.0
        
        # CSV header fields
        self.csv_headers = [
            "time",
            "stride_x", "stride_y", "frequency", "step_height", "body_height", "yaw_rate",
            "roll_cmd", "pitch_cmd", "yaw_cmd",
            "base_x", "base_y", "base_z",
            "roll", "pitch", "yaw",
            "vx", "vy", "vz",
            "wx", "wy", "wz",
            "FL_hip_q", "FL_thigh_q", "FL_calf_q",
            "FR_hip_q", "FR_thigh_q", "FR_calf_q",
            "RL_hip_q", "RL_thigh_q", "RL_calf_q",
            "RR_hip_q", "RR_thigh_q", "RR_calf_q",
            "FL_foot_x", "FL_foot_y", "FL_foot_z",
            "FR_foot_x", "FR_foot_y", "FR_foot_z",
            "RL_foot_x", "RL_foot_y", "RL_foot_z",
            "RR_foot_x", "RR_foot_y", "RR_foot_z",
            "FL_contact", "FR_contact", "RL_contact", "RR_contact",
            "FL_force", "FR_force", "RL_force", "RR_force",
            "stability_margin", "electrical_power", "mechanical_power",
            "FL_slip", "FR_slip", "RL_slip", "RR_slip"
        ]

    def start_session(self):
        self.is_recording = True
        self.session_data = []
        self.start_time = time.time()
        print("[+] Session recording started.")

    def stop_session(self, export_format="csv"):
        """
        Stops active recording session and writes data to disk.
        Returns the path to the exported file.
        """
        if not self.is_recording:
            return None
            
        self.is_recording = False
        if not self.session_data:
            print("[-] Recording stopped (no data logged).")
            return None
            
        # Filename based on timestamp
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"locomotion_session_{timestamp_str}"
        
        export_path = ""
        if export_format.lower() == "csv":
            export_path = os.path.join(self.default_dir, f"{filename}.csv")
            with open(export_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_headers)
                writer.writeheader()
                for row in self.session_data:
                    # Filter keys to match headers exactly
                    filtered_row = {k: v for k, v in row.items() if k in self.csv_headers}
                    writer.writerow(filtered_row)
        else: # JSON format
            export_path = os.path.join(self.default_dir, f"{filename}.json")
            meta = {
                "recorded_at": timestamp_str,
                "total_timesteps": len(self.session_data),
                "duration_seconds": self.session_data[-1]["time"] if self.session_data else 0.0,
                "data": self.session_data
            }
            with open(export_path, 'w') as f:
                json.dump(meta, f, indent=2)
                
        print(f"[+] Locomotion session saved successfully to {export_path}")
        return export_path

    def log_step(self, sim_time, gait_generator, base_pos, base_orient_rpy, base_vel_linear, base_vel_angular, joint_angles, foot_positions, contacts, contact_forces, stability_margin, power_elec, power_mech, slips):
        """
        Logs a single simulation timestep.
        """
        if not self.is_recording:
            return
            
        elapsed_time = sim_time # use simulator internal time
        
        row = {
            "time": elapsed_time,
            
            # Gait Commands
            "stride_x": gait_generator.stride_x,
            "stride_y": gait_generator.stride_y,
            "frequency": gait_generator.frequency,
            "step_height": gait_generator.step_height,
            "body_height": gait_generator.body_height,
            "yaw_rate": gait_generator.yaw_rate,
            "roll_cmd": gait_generator.roll_offset,
            "pitch_cmd": gait_generator.pitch_offset,
            "yaw_cmd": gait_generator.yaw_offset,
            
            # Base Odometry & IMU
            "base_x": base_pos[0],
            "base_y": base_pos[1],
            "base_z": base_pos[2],
            "roll": base_orient_rpy[0],
            "pitch": base_orient_rpy[1],
            "yaw": base_orient_rpy[2],
            "vx": base_vel_linear[0],
            "vy": base_vel_linear[1],
            "vz": base_vel_linear[2],
            "wx": base_vel_angular[0],
            "wy": base_vel_angular[1],
            "wz": base_vel_angular[2],
            
            # Joints
            "FL_hip_q": joint_angles[0], "FL_thigh_q": joint_angles[1], "FL_calf_q": joint_angles[2],
            "FR_hip_q": joint_angles[3], "FR_thigh_q": joint_angles[4], "FR_calf_q": joint_angles[5],
            "RL_hip_q": joint_angles[6], "RL_thigh_q": joint_angles[7], "RL_calf_q": joint_angles[8],
            "RR_hip_q": joint_angles[9], "RR_thigh_q": joint_angles[10], "RR_calf_q": joint_angles[11],
            
            # Feet Cartesian Target Coordinates
            "FL_foot_x": foot_positions['FL'][0], "FL_foot_y": foot_positions['FL'][1], "FL_foot_z": foot_positions['FL'][2],
            "FR_foot_x": foot_positions['FR'][0], "FR_foot_y": foot_positions['FR'][1], "FR_foot_z": foot_positions['FR'][2],
            "RL_foot_x": foot_positions['RL'][0], "RL_foot_y": foot_positions['RL'][1], "RL_foot_z": foot_positions['RL'][2],
            "RR_foot_x": foot_positions['RR'][0], "RR_foot_y": foot_positions['RR'][1], "RR_foot_z": foot_positions['RR'][2],
            
            # Contacts & Forces
            "FL_contact": 1 if contacts['FL'] else 0,
            "FR_contact": 1 if contacts['FR'] else 0,
            "RL_contact": 1 if contacts['RL'] else 0,
            "RR_contact": 1 if contacts['RR'] else 0,
            
            "FL_force": contact_forces['FL'],
            "FR_force": contact_forces['FR'],
            "RL_force": contact_forces['RL'],
            "RR_force": contact_forces['RR'],
            
            # Telemetry Analytics
            "stability_margin": stability_margin,
            "electrical_power": power_elec,
            "mechanical_power": power_mech,
            
            "FL_slip": slips['FL'],
            "FR_slip": slips['FR'],
            "RL_slip": slips['RL'],
            "RR_slip": slips['RR']
        }
        
        self.session_data.append(row)
