from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QGridLayout, QGroupBox, QSlider, QComboBox, 
                             QPushButton, QLabel, QFileDialog, QMessageBox,
                             QTabWidget, QDoubleSpinBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeyEvent

import numpy as np
import os
import mujoco

from simulator.sim_manager import SimulationManager
from gait_engine.gait_generator import GaitGenerator
from analytics.stability import StabilityAnalyzer
from analytics.energy import EnergyEstimator
from analytics.slip_detection import SlipDetector
from analytics.gait_metrics import GaitMetricsTracker
from recording.recorder import LocomotionRecorder
from recording.replay import LocomotionReplayer

from gui.widgets import ScrollingGaitTimeline, TelemetryDashboardPanel
from visualization.live_plots import LivePlotsCanvas

class GaitStudioDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("GaitStudio — Quadruped Locomotion R&D Platform")
        self.resize(1300, 850)
        
        # Stylesheet (Isaac Sim / Boston Dynamics theme)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #16171a;
            }
            QWidget {
                color: #dcdee5;
                font-family: 'Segoe UI', sans-serif;
            }
            QGroupBox {
                background-color: #1a1b1f;
                border: 1px solid #2f3036;
                border-radius: 8px;
                margin-top: 15px;
                font-weight: bold;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding-left: 10px;
                padding-right: 10px;
                color: #8a2be2; /* Purple/Blue accent */
            }
            QSlider::groove:horizontal {
                border: 1px solid #2f3036;
                height: 4px;
                background: #232429;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #8a2be2;
                border: none;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QSlider::handle:horizontal:hover {
                background: #39ff14; /* Neon Green on hover */
            }
            QComboBox {
                background-color: #232429;
                border: 1px solid #2f3036;
                border-radius: 4px;
                padding: 5px;
                color: #ffffff;
                font-size: 11px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QPushButton {
                background-color: #232429;
                border: 1px solid #2f3036;
                border-radius: 6px;
                padding: 8px 15px;
                color: #ffffff;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #8a2be2;
                border-color: #8a2be2;
            }
            QPushButton#btn-record {
                background-color: #1b3d22;
                border-color: #2e5936;
            }
            QPushButton#btn-record:hover {
                background-color: #39ff14;
                color: #16171a;
            }
            QPushButton#btn-stop {
                background-color: #4d1c1c;
                border-color: #732a2a;
            }
            QPushButton#btn-stop:hover {
                background-color: #ff3333;
                color: #ffffff;
            }
            QLabel {
                font-size: 11px;
            }
            QLabel#label-title {
                font-size: 16px;
                font-weight: bold;
                color: #ffffff;
                letter-spacing: 1px;
            }
            QLabel#label-subtitle {
                font-size: 10px;
                color: #7a7c85;
                font-weight: bold;
                letter-spacing: 0.5px;
            }
        """)
        
        # Instantiate locomotion architecture subsystems
        self.sim = SimulationManager(mode="mujoco", terrain_type="flat")
        self.gait_gen = GaitGenerator(hip_positions=self.sim.hip_offsets)
        
        self.stability_analyzer = StabilityAnalyzer()
        self.energy_estimator = EnergyEstimator()
        self.slip_detector = SlipDetector()
        self.metrics_tracker = GaitMetricsTracker()
        
        self.recorder = LocomotionRecorder()
        self.replayer = LocomotionReplayer()
        
        # State tracking and digital filter variables
        self.playback_mode = False
        self.filtered_power_elec = 8.0
        self.filtered_power_mech = 0.0
        
        # Build UI layout
        self.sliders = {}
        self.slider_labels = {}
        self.init_ui()
        
        # Real-time execution timer (60Hz GUI update, simulator sub-stepping handled inside sim.step)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.run_simulation_step)
        self.timer.start(16) # ~60 FPS
        
        # Key capture focus
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)


    def init_ui(self):
        # Main central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        # 1. Header Banner
        header_layout = QHBoxLayout()
        title_v_layout = QVBoxLayout()
        
        title_label = QLabel("GAITSTUDIO  //  QUADRUPED LOCOMOTION R&D PLATFORM")
        title_label.setObjectName("label-title")
        subtitle_label = QLabel("ANALYTICAL KINEMATICS, TELEMETRY METRICS, AND PROCEDURAL GAIT SYNTHESIS")
        subtitle_label.setObjectName("label-subtitle")
        
        title_v_layout.addWidget(title_label)
        title_v_layout.addWidget(subtitle_label)
        
        header_layout.addLayout(title_v_layout)
        header_layout.addStretch()
        
        # Simulation Mode selector
        mode_label = QLabel("SIM ENGINE:")
        self.combo_mode = QComboBox()
        self.combo_mode.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.combo_mode.addItems(["MuJoCo 3D Solver", "Standalone Kinematics"])
        self.combo_mode.currentIndexChanged.connect(self.change_simulation_backend)
        
        header_layout.addWidget(mode_label)
        header_layout.addWidget(self.combo_mode)
        
        main_layout.addLayout(header_layout)
        
        # Horizontal layout: Controls (Left) vs Plots (Right)
        body_layout = QHBoxLayout()
        body_layout.setSpacing(15)
        
        # --- LEFT SIDE: GAIT PARAMETERS & COMMANDS ---
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(10)
        
        # Group 1: Preset selectors
        grp_presets = QGroupBox("Locomotion Presets")
        presets_grid = QGridLayout(grp_presets)
        presets_grid.setSpacing(8)
        
        presets_grid.addWidget(QLabel("Gait Pattern:"), 0, 0)
        self.combo_gait = QComboBox()
        self.combo_gait.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.combo_gait.addItems(["Trot", "Walk", "Pace", "Bound"])
        self.combo_gait.currentTextChanged.connect(self.change_gait_pattern)
        presets_grid.addWidget(self.combo_gait, 0, 1)
        
        presets_grid.addWidget(QLabel("Environment:"), 1, 0)
        env_label = QLabel("Flat Ground")
        env_label.setStyleSheet("color: #39ff14; font-weight: bold;")
        presets_grid.addWidget(env_label, 1, 1)
        
        controls_layout.addWidget(grp_presets)
        
        # Group 2: Locomotion Parameters
        grp_params = QGroupBox("Gait Parameters")
        params_grid = QGridLayout(grp_params)
        params_grid.setSpacing(6)
        
        # Parameter sliders
        # (label, slider_attr, min_val, max_val, default_val, scale)
        slider_configs = [
            ("Stride X (Forward)", "stride_x", -15, 15, 0, 100.0), # -0.15m to 0.15m
            ("Stride Y (Lateral)", "stride_y", -8, 8, 0, 100.0),   # -0.08m to 0.08m
            ("Step Height (Lift)", "step_height", 1, 10, 6, 100.0), # 0.01m to 0.10m
            ("Gait Frequency (Hz)", "frequency", 5, 30, 15, 10.0), # 0.5Hz to 3.0Hz
            ("Body Height (Stand)", "body_height", 18, 35, 25, 100.0), # 0.18m to 0.35m
            ("Yaw Turning Rate", "yaw_rate", -5, 5, 0, 10.0) # -0.5 to 0.5 rad/s
        ]
        
        for idx, (lbl, attr, mn, mx, df, scale) in enumerate(slider_configs):
            lbl_widget = QLabel(f"{lbl}: {df/scale:.2f}")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            slider.setMinimum(mn)
            slider.setMaximum(mx)
            slider.setValue(df)
            slider.valueChanged.connect(lambda val, a=attr, s=scale, l=lbl_widget, name=lbl: self.on_slider_changed(a, s, l, name, val))
            
            params_grid.addWidget(lbl_widget, idx*2, 0)
            params_grid.addWidget(slider, idx*2+1, 0)
            
            self.sliders[attr] = slider
            self.slider_labels[attr] = lbl_widget
            
        controls_layout.addWidget(grp_params)
        
        # Group 3: Attitude Offsets
        grp_attitude = QGroupBox("Torso Attitude Offsets")
        attitude_grid = QGridLayout(grp_attitude)
        attitude_grid.setSpacing(6)
        
        attitude_configs = [
            ("Base Roll (Roll)", "roll_offset", -30, 30, 0, 100.0),  # -0.30 to 0.30 rad
            ("Base Pitch (Pitch)", "pitch_offset", -30, 30, 0, 100.0), # -0.30 to 0.30 rad
            ("Base Yaw (Yaw)", "yaw_offset", -30, 30, 0, 100.0)   # -0.30 to 0.30 rad
        ]
        
        for idx, (lbl, attr, mn, mx, df, scale) in enumerate(attitude_configs):
            lbl_widget = QLabel(f"{lbl}: {df/scale:.2f} rad")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            slider.setMinimum(mn)
            slider.setMaximum(mx)
            slider.setValue(df)
            slider.valueChanged.connect(lambda val, a=attr, s=scale, l=lbl_widget, name=lbl: self.on_slider_changed(a, s, l, name, val))
            
            attitude_grid.addWidget(lbl_widget, idx*2, 0)
            attitude_grid.addWidget(slider, idx*2+1, 0)
            
            self.sliders[attr] = slider
            self.slider_labels[attr] = lbl_widget
            
        controls_layout.addWidget(grp_attitude)
        
        # Group 4: Recording Subsystem Controls
        grp_recording = QGroupBox("Locomotion Logging")
        rec_layout = QVBoxLayout(grp_recording)
        rec_layout.setSpacing(8)
        
        rec_btns = QHBoxLayout()
        self.btn_start = QPushButton("RECORD")
        self.btn_start.setObjectName("btn-record")
        self.btn_start.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_start.clicked.connect(self.start_recording)
        
        self.btn_stop = QPushButton("STOP & EXPORT")
        self.btn_stop.setObjectName("btn-stop")
        self.btn_stop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_recording)
        
        rec_btns.addWidget(self.btn_start)
        rec_btns.addWidget(self.btn_stop)
        rec_layout.addLayout(rec_btns)
        
        # Playback section
        playback_layout = QHBoxLayout()
        self.combo_format = QComboBox()
        self.combo_format.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.combo_format.addItems(["CSV Trajectory Sheets", "JSON Performance Frame"])
        
        self.btn_replay = QPushButton("REPLAY TRIAL")
        self.btn_replay.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_replay.clicked.connect(self.toggle_replay)
        
        playback_layout.addWidget(self.combo_format)
        playback_layout.addWidget(self.btn_replay)
        rec_layout.addLayout(playback_layout)
        
        controls_layout.addWidget(grp_recording)
        
        # Set controls width
        controls_container = QWidget()
        controls_container.setLayout(controls_layout)
        controls_container.setFixedWidth(280)
        
        body_layout.addWidget(controls_container)
        
        # --- RIGHT SIDE: TELEMETRY PLOTS & REAL-TIME DIAGRAMS ---
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)
        
        # 1. Telemetry Dashboard (Symmetry, Power, Stability margin)
        self.telemetry_panel = TelemetryDashboardPanel()
        right_layout.addWidget(self.telemetry_panel)
        
        # 2. Gantt Scrolling Timeline
        grp_gantt = QGroupBox("Gait Phase Timeline (Neon Green = Stance, Dark = Swing)")
        gantt_layout = QVBoxLayout(grp_gantt)
        gantt_layout.setContentsMargins(5, 10, 5, 5)
        self.timeline_widget = ScrollingGaitTimeline()
        gantt_layout.addWidget(self.timeline_widget)
        right_layout.addWidget(grp_gantt)
        
        # 3. Matplotlib live plots canvas
        grp_plots = QGroupBox("Dynamic State Telemetry & Stability Analytics")
        plots_layout = QVBoxLayout(grp_plots)
        plots_layout.setContentsMargins(5, 5, 5, 5)
        self.plots_canvas = LivePlotsCanvas()
        plots_layout.addWidget(self.plots_canvas)
        right_layout.addWidget(grp_plots)
        
        body_layout.addLayout(right_layout)
        
        main_layout.addLayout(body_layout)
        
        # Teleoperation instructions at footer
        footer_label = QLabel("CONTROLS:  [↑/↓] Forward/Back  |  [←/→] Turn Left/Right  |  [PgUp/PgDn] Lateral  |  [Home/End] Roll  |  [Ins/Del] Pitch  |  [Space] E-Brake  |  [WASD] MuJoCo Camera")
        footer_label.setStyleSheet("color: #7a7c85; font-size: 10px; font-weight: bold; letter-spacing: 0.5px;")
        main_layout.addWidget(footer_label)

    def on_slider_changed(self, attr, scale, label_widget, display_name, val):
        scaled_val = val / scale
        
        # Update Gait Generator or Local value
        setattr(self.gait_gen, attr, scaled_val)
        
        # Update label
        if "rad" in label_widget.text() or "offset" in attr:
            label_widget.setText(f"{display_name}: {scaled_val:.2f} rad")
        else:
            label_widget.setText(f"{display_name}: {scaled_val:.2f}")

    def change_gait_pattern(self, name):
        self.gait_gen.set_gait(name.lower())
        print(f"[+] Gait changed to: {name.upper()}")
        
        # Update Stance Width or nominal settings if needed
        # We can dynamically reset/adjust sliders if gait is walking (slower frequency, larger duty factor)
        if name.lower() == "walk":
            self.sliders["frequency"].setValue(10) # 1.0 Hz
            self.sliders["step_height"].setValue(4) # 0.04m step height
        else:
            self.sliders["frequency"].setValue(15) # 1.5 Hz
            self.sliders["step_height"].setValue(6) # 0.06m step height



    def change_simulation_backend(self, idx):
        # Toggle between MuJoCo and Standalone
        mode_key = "mujoco" if idx == 0 else "standalone"
        
        # Close old simulator
        self.sim.close()
        
        # Start new simulator (flat ground only)
        self.sim = SimulationManager(mode=mode_key, terrain_type="flat")
        self.gait_gen.hip_positions = self.sim.hip_offsets
        
        # Deep state resets to prevent startup control shocks
        self.gait_gen.reset()
        self.metrics_tracker.reset()
        self.energy_estimator.reset()
        
        # If playback is active, disable it
        if self.playback_mode:
            self.toggle_replay()

    def start_recording(self):
        self.recorder.start_session()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_start.setText("RECORDING...")

    def stop_recording(self):
        # Determine format
        fmt = "csv" if self.combo_format.currentIndex() == 0 else "json"
        filepath = self.recorder.stop_session(export_format=fmt)
        
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_start.setText("RECORD")
        
        if filepath:
            QMessageBox.information(self, "Telemetry Logged", f"Locomotion trial successfully saved to:\n\n{os.path.basename(filepath)}")

    def toggle_replay(self):
        if self.playback_mode:
            self.playback_mode = False
            self.btn_replay.setText("REPLAY TRIAL")
            self.btn_replay.setStyleSheet("")
            print("[*] Replay terminated. Resuming live simulation.")
        else:
            # File Dialog to select CSV file
            file_path, _ = QFileDialog.getOpenFileName(self, "Load Locomotion Trial Sheet", "recording/sessions", "CSV Sheets (*.csv)")
            if file_path:
                ok = self.replayer.load_session(file_path)
                if ok:
                    self.playback_mode = True
                    self.btn_replay.setText("STOP PLAYBACK")
                    self.btn_replay.setStyleSheet("background-color: #4a2b91; border-color: #6a3bbb;")
                else:
                    QMessageBox.warning(self, "Loading Failed", "Failed to parse locomotion file.")

    def run_simulation_step(self):
        """
        Executes one loop of the control dashboard.
        In normal mode: updates master phase clock, solves IK, steps simulator, logs telemetry, updates GUI.
        In playback mode: loads frame from replayer, updates joints in simulator directly, updates telemetry.
        """
        # Process any pending keys from the MuJoCo viewer thread safely
        if hasattr(self.sim, 'pending_keys') and self.sim.pending_keys:
            keys_to_process = list(self.sim.pending_keys)
            self.sim.pending_keys.clear()
            
            glfw_to_qt = {
                265: Qt.Key.Key_Up,
                264: Qt.Key.Key_Down,
                263: Qt.Key.Key_Left,
                262: Qt.Key.Key_Right,
                266: Qt.Key.Key_PageUp,
                267: Qt.Key.Key_PageDown,
                268: Qt.Key.Key_Home,
                269: Qt.Key.Key_End,
                260: Qt.Key.Key_Insert,
                261: Qt.Key.Key_Delete,
                32: Qt.Key.Key_Space
            }
            
            for keycode in keys_to_process:
                if keycode in glfw_to_qt:
                    qt_key = glfw_to_qt[keycode]
                    event = QKeyEvent(QKeyEvent.Type.KeyPress, qt_key, Qt.KeyboardModifier.NoModifier)
                    self.keyPressEvent(event)

        if self.playback_mode:
            # --- PLAYBACK MODE ---
            frame = self.replayer.get_next_frame()
            if frame is None:
                self.toggle_replay() # End of file
                return
                
            # Feed replayed joint states directly to simulator base and joints
            if self.sim.mode == "standalone":
                self.sim.sa_base_pos = np.array([frame["base_x"], frame["base_y"], frame["base_z"]])
                self.sim.sa_base_rpy = np.array([frame["roll"], frame["pitch"], frame["yaw"]])
                self.sim.state['joint_q'] = np.array([
                    frame["FL_hip_q"], frame["FL_thigh_q"], frame["FL_calf_q"],
                    frame["FR_hip_q"], frame["FR_thigh_q"], frame["FR_calf_q"],
                    frame["RL_hip_q"], frame["RL_thigh_q"], frame["RL_calf_q"],
                    frame["RR_hip_q"], frame["RR_thigh_q"], frame["RR_calf_q"]
                ])
                contacts = {
                    'FL': bool(frame["FL_contact"]), 'FR': bool(frame["FR_contact"]),
                    'RL': bool(frame["RL_contact"]), 'RR': bool(frame["RR_contact"])
                }
                self.sim.state['foot_contacts'] = contacts
                self.sim.state['foot_forces'] = {
                    'FL': frame["FL_force"], 'FR': frame["FR_force"],
                    'RL': frame["RL_force"], 'RR': frame["RR_force"]
                }
                self.sim.state['base_vel_lin'] = np.array([frame["vx"], frame["vy"], frame["vz"]])
                self.sim.state['base_vel_ang'] = np.array([frame["wx"], frame["wy"], frame["wz"]])
            else:
                self.sim.mj_data.qpos[0:3] = [frame["base_x"], frame["base_y"], frame["base_z"]]
                cr, sr = np.cos(frame["roll"]/2), np.sin(frame["roll"]/2)
                cp, sp = np.cos(frame["pitch"]/2), np.sin(frame["pitch"]/2)
                cy, sy = np.cos(frame["yaw"]/2), np.sin(frame["yaw"]/2)
                w = cr*cp*cy + sr*sp*sy
                x = sr*cp*cy - cr*sp*sy
                y = cr*sp*cy + sr*cp*sy
                z = cr*cp*sy - sr*sp*cy
                self.sim.mj_data.qpos[3:7] = [w, x, y, z]
                
                joints = [
                    "FL_hip_q", "FL_thigh_q", "FL_calf_q",
                    "FR_hip_q", "FR_thigh_q", "FR_calf_q",
                    "RL_hip_q", "RL_thigh_q", "RL_calf_q",
                    "RR_hip_q", "RR_thigh_q", "RR_calf_q"
                ]
                for idx, j_name in enumerate(joints):
                    self.sim.mj_data.qpos[7 + idx] = frame[j_name]
                    
                mujoco.mj_forward(self.sim.mj_model, self.sim.mj_data)
                if self.sim.mj_viewer is not None:
                    self.sim.mj_viewer.sync()
            
            self.sim.sim_time = frame["time"]
            stability_margin = frame["stability_margin"]
            safety_state = "stable" if stability_margin > 0.03 else ("warning" if stability_margin > 0.0 else "unstable")
            
            stability_info = {
                'support_polygon': [[frame["FL_foot_x"] + 0.185, frame["FL_foot_y"] + 0.10] if frame["FL_contact"] else None,
                                    [frame["FR_foot_x"] + 0.185, frame["FR_foot_y"] - 0.10] if frame["FR_contact"] else None,
                                    [frame["RL_foot_x"] - 0.185, frame["RL_foot_y"] + 0.10] if frame["RL_contact"] else None,
                                    [frame["RR_foot_x"] - 0.185, frame["RR_foot_y"] - 0.10] if frame["RR_contact"] else None],
                'state': safety_state
            }
            stability_info['support_polygon'] = [p for p in stability_info['support_polygon'] if p is not None]
            
            contact_forces = {
                'FL': frame["FL_force"], 'FR': frame["FR_force"],
                'RL': frame["RL_force"], 'RR': frame["RR_force"]
            }
            contacts = {k: f > 0.1 for k, f in contact_forces.items()}
            
            # Apply light smoothing during playback to clean up recorded noise
            alpha_filter = 0.15
            self.filtered_power_elec = alpha_filter * frame["electrical_power"] + (1.0 - alpha_filter) * self.filtered_power_elec
            self.filtered_power_mech = alpha_filter * frame["mechanical_power"] + (1.0 - alpha_filter) * self.filtered_power_mech
            
            power_elec = self.filtered_power_elec
            power_mech = self.filtered_power_mech
            symmetry_pct = frame["stride_x"] * 100.0 if "stride_x" in frame else 100.0
            
        else:
            # --- NORMAL SIMULATION MODE ---
            substeps = 4
            
            # Temporary accumulators for substep averaging to resolve aliasing
            accum_stability_margin = 0.0
            accum_power_elec = 0.0
            accum_power_mech = 0.0
            accum_symmetry = 0.0
            accum_forces = {'FL': 0.0, 'FR': 0.0, 'RL': 0.0, 'RR': 0.0}
            accum_contacts = {'FL': 0.0, 'FR': 0.0, 'RL': 0.0, 'RR': 0.0}
            
            # Trace support polygon for the last step in the sequence
            last_stability_info = None
            
            for _ in range(substeps):
                # 1. Increment gait generator phase clock (240Hz)
                self.gait_gen.update_phase(self.sim.dt)
                
                # 2. Compute 3D target coordinates for foot tips relative to hips
                foot_targets = self.gait_gen.compute_foot_targets()
                
                # 3. Step simulator by 1 physics step of self.sim.dt (240Hz)
                self.sim.step_single(foot_targets, self.gait_gen)
                
                # 4. Extract telemetry and run Locomotion Analytics at 240Hz
                com_xy = self.sim.state['base_pos'][:2]
                contacts = self.sim.state['foot_contacts']
                contact_forces = self.sim.state['foot_forces']
                
                # Get body rotation matrix for slip estimator frame alignment
                R_body = self.gait_gen.get_rotation_matrix(
                    self.sim.state['base_rpy'][0],
                    self.sim.state['base_rpy'][1],
                    self.sim.state['base_rpy'][2]
                )
                
                # FIXED: Stability calculation in body-local coordinate frame.
                # Projecting local relative foot targets keeps the support polygon 
                # locked to the axis-aligned torso footprint, resolving yaw drift errors.
                foot_xy_contacts = {}
                for leg in ['FL', 'FR', 'RL', 'RR']:
                    pos_local = self.sim.state['foot_positions_hip'][leg] + self.sim.hip_offsets[leg]
                    foot_xy_contacts[leg] = (pos_local[:2], contacts[leg])
                    
                com_local = np.array([0.0, 0.0])
                stability_info = self.stability_analyzer.compute_stability(com_local, foot_xy_contacts)
                last_stability_info = stability_info
                
                # energy
                power_info = self.energy_estimator.compute_power(self.sim.state['joint_tau'], self.sim.state['joint_dq'])
                
                # slip detector with proper world frame alignment
                slips = {}
                for leg in ['FL', 'FR', 'RL', 'RR']:
                    l_idx = ['FL', 'FR', 'RL', 'RR'].index(leg)
                    q_dot = self.sim.state['joint_dq'][l_idx*3 : l_idx*3+3]
                    foot_pos_body = self.sim.state['foot_positions_hip'][leg] + self.sim.hip_offsets[leg]
                    
                    slip_info = self.slip_detector.detect_slip(
                        leg, contacts[leg], q_dot, 
                        self.sim.state['base_vel_lin'], self.sim.state['base_vel_ang'], 
                        foot_pos_body, R_body=R_body
                    )
                    slips[leg] = slip_info['slip_velocity']
                    
                # metrics tracker at 240Hz
                self.metrics_tracker.update(contacts, [self.gait_gen.stride_x, self.gait_gen.stride_y, self.gait_gen.yaw_rate], self.sim.state['base_vel_lin'], self.sim.dt)
                metrics = self.metrics_tracker.get_metrics()
                
                # Accumulate values
                accum_stability_margin += stability_info['stability_margin']
                accum_power_elec += power_info['electrical_power']
                accum_power_mech += power_info['mechanical_power']
                accum_symmetry += metrics['gait_symmetry'] * 100.0
                for leg in ['FL', 'FR', 'RL', 'RR']:
                    accum_forces[leg] += contact_forces[leg]
                    accum_contacts[leg] += 1.0 if contacts[leg] else 0.0
                
                # 5. Log step in session recorder at 240Hz
                if self.recorder.is_recording:
                    self.recorder.log_step(
                        self.sim.sim_time, self.gait_gen, 
                        self.sim.state['base_pos'], self.sim.state['base_rpy'], 
                        self.sim.state['base_vel_lin'], self.sim.state['base_vel_ang'],
                        self.sim.state['joint_q'], self.gait_gen.compute_foot_targets(),
                        contacts, contact_forces, stability_info['stability_margin'], 
                        power_info['electrical_power'], power_info['mechanical_power'], slips
                    )
            
            # Compute raw averages over the step window
            raw_power_elec = accum_power_elec / substeps
            raw_power_mech = accum_power_mech / substeps
            
            # FIXED: Apply first-order low-pass filter (Exponential Moving Average) to the power signal
            # to remove high-frequency physical impact noise.
            alpha_filter = 0.15
            self.filtered_power_elec = alpha_filter * raw_power_elec + (1.0 - alpha_filter) * self.filtered_power_elec
            self.filtered_power_mech = alpha_filter * raw_power_mech + (1.0 - alpha_filter) * self.filtered_power_mech
            
            stability_margin = accum_stability_margin / substeps
            safety_state = "stable" if stability_margin > 0.03 else ("warning" if stability_margin > 0.0 else "unstable")
            power_elec = self.filtered_power_elec
            power_mech = self.filtered_power_mech
            symmetry_pct = accum_symmetry / substeps
            
            contact_forces = {k: v / substeps for k, v in accum_forces.items()}
            contacts = {k: (v / substeps) > 0.5 for k, v in accum_contacts.items()} # Majority vote for logical state
            stability_info = last_stability_info
            stability_info['stability_margin'] = stability_margin
            stability_info['state'] = safety_state
                
        # --- UPDATE GUI TELEMETRY WIDGETS ---
        self.telemetry_panel.update_telemetry(stability_margin, safety_state, power_elec, symmetry_pct)
        self.timeline_widget.update_states(contacts)
        self.plots_canvas.update_plots(np.array([0.0, 0.0]), stability_info, contact_forces, power_elec, power_mech, self.sim.dt)


    def keyPressEvent(self, event: QKeyEvent):
        """
        Keyboard teleoperation handler using arrow keys and auxiliary keys.
        WASD keys are left free for MuJoCo's built-in camera controls.
        
        Arrow Up/Down    = Stride X (forward/backward)
        Arrow Left/Right = Yaw turning (left/right)
        Page Up/Down     = Stride Y (lateral left/right)
        Home/End         = Attitude Roll
        Insert/Delete    = Attitude Pitch
        Space            = Emergency stop (zero all)
        """
        # --- Locomotion movement ---
        if event.key() == Qt.Key.Key_Up:
            # Stride X forward
            val = self.sliders["stride_x"].value() + 1
            self.sliders["stride_x"].setValue(int(np.clip(val, -15, 15)))
        elif event.key() == Qt.Key.Key_Down:
            # Stride X backward
            val = self.sliders["stride_x"].value() - 1
            self.sliders["stride_x"].setValue(int(np.clip(val, -15, 15)))
            
        elif event.key() == Qt.Key.Key_Left:
            # Yaw turn counter-clockwise (turn left)
            val = self.sliders["yaw_rate"].value() + 1
            self.sliders["yaw_rate"].setValue(int(np.clip(val, -5, 5)))
        elif event.key() == Qt.Key.Key_Right:
            # Yaw turn clockwise (turn right)
            val = self.sliders["yaw_rate"].value() - 1
            self.sliders["yaw_rate"].setValue(int(np.clip(val, -5, 5)))
            
        # --- Lateral stride ---
        elif event.key() == Qt.Key.Key_PageUp:
            # Stride Y left
            val = self.sliders["stride_y"].value() + 1
            self.sliders["stride_y"].setValue(int(np.clip(val, -8, 8)))
        elif event.key() == Qt.Key.Key_PageDown:
            # Stride Y right
            val = self.sliders["stride_y"].value() - 1
            self.sliders["stride_y"].setValue(int(np.clip(val, -8, 8)))
            
        # --- Attitude controls ---
        elif event.key() == Qt.Key.Key_Home:
            # Roll left
            val = self.sliders["roll_offset"].value() - 2
            self.sliders["roll_offset"].setValue(int(np.clip(val, -30, 30)))
        elif event.key() == Qt.Key.Key_End:
            # Roll right
            val = self.sliders["roll_offset"].value() + 2
            self.sliders["roll_offset"].setValue(int(np.clip(val, -30, 30)))
            
        elif event.key() == Qt.Key.Key_Insert:
            # Pitch forward
            val = self.sliders["pitch_offset"].value() + 2
            self.sliders["pitch_offset"].setValue(int(np.clip(val, -30, 30)))
        elif event.key() == Qt.Key.Key_Delete:
            # Pitch backward
            val = self.sliders["pitch_offset"].value() - 2
            self.sliders["pitch_offset"].setValue(int(np.clip(val, -30, 30)))
            
        # --- Emergency stop ---
        elif event.key() == Qt.Key.Key_Space:
            # Stop all motion (brake)
            self.sliders["stride_x"].setValue(0)
            self.sliders["stride_y"].setValue(0)
            self.sliders["yaw_rate"].setValue(0)
            self.sliders["roll_offset"].setValue(0)
            self.sliders["pitch_offset"].setValue(0)
            self.sliders["yaw_offset"].setValue(0)
            print("[*] E-Brake: Locomotion vectors zeroed.")
            
        else:
            # Bubble up standard key press events
            super().keyPressEvent(event)

    def closeEvent(self, event):
        # Shut down simulator when closing window
        self.sim.close()
        event.accept()

    def mousePressEvent(self, event):
        # Focus main window when clicked anywhere on the background
        self.setFocus()
        super().mousePressEvent(event)
