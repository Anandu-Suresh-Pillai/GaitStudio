from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QGridLayout, QGroupBox, QSlider, QComboBox, 
                             QPushButton, QLabel, QFileDialog, QMessageBox,
                             QTabWidget, QDoubleSpinBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeyEvent

import numpy as np
import os
try:
    import mujoco
except ImportError:
    mujoco = None

from simulator.sim_manager import SimulationManager, QUADRUPED_PRESETS
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
        # Uses the default Custom configuration on startup
        default_config = QUADRUPED_PRESETS["Custom (User-Defined)"]
        self.sim = SimulationManager(mode="mujoco", terrain_type="flat", config=default_config)
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
        
        # Horizontal layout: Left Tab Control Panel vs Right Plots Dashboard
        body_layout = QHBoxLayout()
        body_layout.setSpacing(15)
        
        # --- LEFT SIDE: TAB CONTEXT (CONTROL vs HARDWARE) ---
        self.tabs = QTabWidget()
        self.tabs.setFixedWidth(280)
        self.tabs.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #2f3036;
                background-color: #16171a;
                border-radius: 6px;
            }
            QTabBar::tab {
                background: #1a1b1f;
                border: 1px solid #2f3036;
                padding: 6px 12px;
                font-size: 10px;
                font-weight: bold;
                color: #7a7c85;
            }
            QTabBar::tab:selected {
                background: #232429;
                color: #ffffff;
                border-bottom-color: #232429;
            }
        """)
        
        # TAB 1 Layout: Locomotion & Teleoperation Controls
        tab_loco_widget = QWidget()
        tab_loco_layout = QVBoxLayout(tab_loco_widget)
        tab_loco_layout.setContentsMargins(5, 5, 5, 5)
        tab_loco_layout.setSpacing(10)
        
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
        
        tab_loco_layout.addWidget(grp_presets)
        
        # Group 2: Locomotion Parameters
        grp_params = QGroupBox("Gait Parameters")
        params_grid = QGridLayout(grp_params)
        params_grid.setSpacing(6)
        
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
            
        tab_loco_layout.addWidget(grp_params)
        
        # Group 3: Torso Attitude Offsets
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
            
        tab_loco_layout.addWidget(grp_attitude)
        
        # Group 4: Recording Subsystem
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
        
        tab_loco_layout.addWidget(grp_recording)
        tab_loco_layout.addStretch()
        
        self.tabs.addTab(tab_loco_widget, "CONTROL")
        
        # TAB 2 Layout: Physical Hardware & presets
        tab_hw_widget = QWidget()
        tab_hw_layout = QVBoxLayout(tab_hw_widget)
        tab_hw_layout.setContentsMargins(5, 5, 5, 5)
        tab_hw_layout.setSpacing(10)
        
        # Group: Hardware Presets selector
        grp_hw_presets = QGroupBox("Hardware Presets")
        hw_presets_layout = QVBoxLayout(grp_hw_presets)
        
        self.combo_robot = QComboBox()
        self.combo_robot.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.combo_robot.addItems(list(QUADRUPED_PRESETS.keys()))
        self.combo_robot.setCurrentText("Custom (User-Defined)")
        self.combo_robot.currentTextChanged.connect(self.on_robot_preset_changed)
        hw_presets_layout.addWidget(self.combo_robot)
        tab_hw_layout.addWidget(grp_hw_presets)
        
        # Group: Physical dimensions
        grp_dims = QGroupBox("Link & Torso Dimensions")
        dims_grid = QGridLayout(grp_dims)
        dims_grid.setSpacing(6)
        
        self.spin_base_mass = self.create_spinbox(1.0, 50.0, 10.0, dims_grid, "Torso Mass (kg):", 0)
        self.spin_hip_l = self.create_spinbox(0.01, 0.30, 0.06, dims_grid, "Hip Link (m):", 1)
        self.spin_thigh_l = self.create_spinbox(0.05, 0.60, 0.22, dims_grid, "Thigh Link (m):", 2)
        self.spin_calf_l = self.create_spinbox(0.05, 0.60, 0.22, dims_grid, "Calf Link (m):", 3)
        self.spin_base_length = self.create_spinbox(0.10, 1.20, 0.45, dims_grid, "Torso Length (m):", 4)
        self.spin_base_width = self.create_spinbox(0.05, 0.80, 0.20, dims_grid, "Torso Width (m):", 5)
        
        tab_hw_layout.addWidget(grp_dims)
        
        # Group: Actuator Configuration
        grp_elec = QGroupBox("Actuator Electrical Specs")
        elec_grid = QGridLayout(grp_elec)
        elec_grid.setSpacing(6)
        
        self.spin_gear = self.create_spinbox(1.0, 50.0, 10.0, elec_grid, "Gear Reduction:", 0)
        self.spin_kt = self.create_spinbox(0.01, 2.0, 0.18, elec_grid, "Torque Const (Kt):", 1)
        self.spin_r_coil = self.create_spinbox(0.01, 5.0, 0.20, elec_grid, "Coil Resist (R):", 2)
        self.spin_kp = self.create_spinbox(50.0, 2000.0, 500.0, elec_grid, "Joint Stiffness (Kp):", 3)
        
        tab_hw_layout.addWidget(grp_elec)
        
        # Re-initialize Apply Button
        self.btn_apply_hw = QPushButton("APPLY HARDWARE CONFIG")
        self.btn_apply_hw.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_apply_hw.setStyleSheet("background-color: #4a2b91; border-color: #6a3bbb;")
        self.btn_apply_hw.clicked.connect(self.apply_custom_hardware_config)
        tab_hw_layout.addWidget(self.btn_apply_hw)
        
        tab_hw_layout.addStretch()
        self.tabs.addTab(tab_hw_widget, "HARDWARE")
        
        body_layout.addWidget(self.tabs)
        
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

    def create_spinbox(self, min_val, max_val, default_val, layout, label_text, row):
        """
        Creates styled spinboxes for numeric hardware adjustment.
        """
        layout.addWidget(QLabel(label_text), row, 0)
        spin = QDoubleSpinBox()
        spin.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        spin.setRange(min_val, max_val)
        spin.setSingleStep(0.01 if max_val <= 5.0 else 10.0)
        spin.setValue(default_val)
        spin.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #232429;
                border: 1px solid #2f3036;
                color: #ffffff;
                padding: 3px;
                border-radius: 4px;
                font-family: Consolas;
            }
        """)
        layout.addWidget(spin, row, 1)
        return spin

    def on_slider_changed(self, attr, scale, label_widget, display_name, val):
        scaled_val = val / scale
        setattr(self.gait_gen, attr, scaled_val)
        if "rad" in label_widget.text() or "offset" in attr:
            label_widget.setText(f"{display_name}: {scaled_val:.2f} rad")
        else:
            label_widget.setText(f"{display_name}: {scaled_val:.2f}")

    def change_gait_pattern(self, name):
        self.gait_gen.set_gait(name.lower())
        print(f"[+] Gait changed to: {name.upper()}")
        if name.lower() == "walk":
            self.sliders["frequency"].setValue(10) # 1.0 Hz
            self.sliders["step_height"].setValue(4) # 0.04m step height
        else:
            self.sliders["frequency"].setValue(15) # 1.5 Hz
            self.sliders["step_height"].setValue(6) # 0.06m step height

    def on_robot_preset_changed(self, preset_name):
        """
        Fills out double-spin inputs when preset is triggered.
        """
        config = QUADRUPED_PRESETS[preset_name]
        
        self.spin_base_mass.setValue(config["base_mass"])
        self.spin_hip_l.setValue(config["hip_l"])
        self.spin_thigh_l.setValue(config["thigh_l"])
        self.spin_calf_l.setValue(config["calf_l"])
        self.spin_base_length.setValue(config["base_dim"][0])
        self.spin_base_width.setValue(config["base_dim"][1])
        
        self.spin_gear.setValue(config["gear_ratio"])
        self.spin_kt.setValue(config["k_t"])
        self.spin_r_coil.setValue(config["r_coil"])
        self.spin_kp.setValue(config["kp_joint"])
        
        if preset_name != "Custom (User-Defined)":
            self.apply_custom_hardware_config()

    def apply_custom_hardware_config(self):
        """
        Recompiles physics environment dynamically with active numeric spinbox values.
        """
        config = {
            "base_mass": self.spin_base_mass.value(),
            "base_dim": [self.spin_base_length.value(), self.spin_base_width.value(), 0.10],
            "hip_l": self.spin_hip_l.value(),
            "thigh_l": self.spin_thigh_l.value(),
            "calf_l": self.spin_calf_l.value(),
            "gear_ratio": self.spin_gear.value(),
            "k_t": self.spin_kt.value(),
            "r_coil": self.spin_r_coil.value(),
            "kp_joint": self.spin_kp.value()
        }
        
        print("[+] Commencing customized quadruped hardware synthesis...")
        
        # 1. Close current session
        self.sim.close()
        
        # 2. Rebuild physics with the requested specs
        mode_idx = self.combo_mode.currentIndex()
        mode_key = "mujoco" if mode_idx == 0 else "standalone"
        self.sim = SimulationManager(mode=mode_key, terrain_type="flat", config=config)
        
        # 3. Align kinematics frames
        self.gait_gen.hip_positions = self.sim.hip_offsets
        self.gait_gen.body_height = config["thigh_l"] + config["calf_l"] * 0.8
        self.sliders["body_height"].setValue(int(self.gait_gen.body_height * 100))
        
        # 4. Synchronize hardware analyzer parameters
        self.energy_estimator.gear_ratio = config["gear_ratio"]
        self.energy_estimator.k_t = config["k_t"]
        self.energy_estimator.r_coil = config["r_coil"]
        
        # 5. Reset analysis streams
        self.gait_gen.reset()
        self.metrics_tracker.reset()
        self.energy_estimator.reset()
        
        print("[+] Quadruped compiled and successfully instantiated.")

    def change_simulation_backend(self, idx):
        mode_key = "mujoco" if idx == 0 else "standalone"
        self.sim.close()
        
        # Fetch current spinbox settings to keep the same hardware configuration
        config = {
            "base_mass": self.spin_base_mass.value(),
            "base_dim": [self.spin_base_length.value(), self.spin_base_width.value(), 0.10],
            "hip_l": self.spin_hip_l.value(),
            "thigh_l": self.spin_thigh_l.value(),
            "calf_l": self.spin_calf_l.value(),
            "gear_ratio": self.spin_gear.value(),
            "k_t": self.spin_kt.value(),
            "r_coil": self.spin_r_coil.value(),
            "kp_joint": self.spin_kp.value()
        }
        
        self.sim = SimulationManager(mode=mode_key, terrain_type="flat", config=config)
        self.gait_gen.hip_positions = self.sim.hip_offsets
        
        self.gait_gen.reset()
        self.metrics_tracker.reset()
        self.energy_estimator.reset()
        
        if self.playback_mode:
            self.toggle_replay()

    def start_recording(self):
        self.recorder.start_session()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_start.setText("RECORDING...")

    def stop_recording(self):
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
        """
        if hasattr(self.sim, 'pending_keys') and self.sim.pending_keys:
            keys_to_process = list(self.sim.pending_keys)
            self.sim.pending_keys.clear()
            
            glfw_to_qt = {
                265: Qt.Key.Key_Up, 264: Qt.Key.Key_Down,
                263: Qt.Key.Key_Left, 262: Qt.Key.Key_Right,
                266: Qt.Key.Key_PageUp, 267: Qt.Key.Key_PageDown,
                268: Qt.Key.Key_Home, 269: Qt.Key.Key_End,
                260: Qt.Key.Key_Insert, 261: Qt.Key.Key_Delete,
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
                self.toggle_replay()
                return
                
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
            com_xy = np.array([frame["base_x"], frame["base_y"]])
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
            
            alpha_filter = 0.15
            self.filtered_power_elec = alpha_filter * frame["electrical_power"] + (1.0 - alpha_filter) * self.filtered_power_elec
            self.filtered_power_mech = alpha_filter * frame["mechanical_power"] + (1.0 - alpha_filter) * self.filtered_power_mech
            
            power_elec = self.filtered_power_elec
            power_mech = self.filtered_power_mech
            symmetry_pct = frame["stride_x"] * 100.0 if "stride_x" in frame else 100.0
            
        else:
            # --- NORMAL SIMULATION MODE ---
            substeps = 4
            
            accum_stability_margin = 0.0
            accum_power_elec = 0.0
            accum_power_mech = 0.0
            accum_symmetry = 0.0
            accum_forces = {'FL': 0.0, 'FR': 0.0, 'RL': 0.0, 'RR': 0.0}
            accum_contacts = {'FL': 0.0, 'FR': 0.0, 'RL': 0.0, 'RR': 0.0}
            
            last_stability_info = None
            
            for _ in range(substeps):
                self.gait_gen.update_phase(self.sim.dt)
                foot_targets = self.gait_gen.compute_foot_targets()
                self.sim.step_single(foot_targets, self.gait_gen)
                
                com_xy = self.sim.state['base_pos'][:2]
                contacts = self.sim.state['foot_contacts']
                contact_forces = self.sim.state['foot_forces']
                
                R_body = self.gait_gen.get_rotation_matrix(
                    self.sim.state['base_rpy'][0],
                    self.sim.state['base_rpy'][1],
                    self.sim.state['base_rpy'][2]
                )
                
                # Stability footprint evaluated in the body-local frame
                foot_xy_contacts = {}
                for leg in ['FL', 'FR', 'RL', 'RR']:
                    pos_local = self.sim.state['foot_positions_hip'][leg] + self.sim.hip_offsets[leg]
                    foot_xy_contacts[leg] = (pos_local[:2], contacts[leg])
                    
                com_local = np.array([0.0, 0.0])
                stability_info = self.stability_analyzer.compute_stability(com_local, foot_xy_contacts)
                last_stability_info = stability_info
                
                power_info = self.energy_estimator.compute_power(self.sim.state['joint_tau'], self.sim.state['joint_dq'])
                
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
                    
                self.metrics_tracker.update(contacts, [self.gait_gen.stride_x, self.gait_gen.stride_y, self.gait_gen.yaw_rate], self.sim.state['base_vel_lin'], self.sim.dt)
                metrics = self.metrics_tracker.get_metrics()
                
                accum_stability_margin += stability_info['stability_margin']
                accum_power_elec += power_info['electrical_power']
                accum_power_mech += power_info['mechanical_power']
                accum_symmetry += metrics['gait_symmetry'] * 100.0
                for leg in ['FL', 'FR', 'RL', 'RR']:
                    accum_forces[leg] += contact_forces[leg]
                    accum_contacts[leg] += 1.0 if contacts[leg] else 0.0
                
                if self.recorder.is_recording:
                    self.recorder.log_step(
                        self.sim.sim_time, self.gait_gen, 
                        self.sim.state['base_pos'], self.sim.state['base_rpy'], 
                        self.sim.state['base_vel_lin'], self.sim.state['base_vel_ang'],
                        self.sim.state['joint_q'], self.gait_gen.compute_foot_targets(),
                        contacts, contact_forces, stability_info['stability_margin'], 
                        power_info['electrical_power'], power_info['mechanical_power'], slips
                    )
            
            raw_power_elec = accum_power_elec / substeps
            raw_power_mech = accum_power_mech / substeps
            
            # EMA Low-pass filters
            alpha_filter = 0.15
            self.filtered_power_elec = alpha_filter * raw_power_elec + (1.0 - alpha_filter) * self.filtered_power_elec
            self.filtered_power_mech = alpha_filter * raw_power_mech + (1.0 - alpha_filter) * self.filtered_power_mech
            
            stability_margin = accum_stability_margin / substeps
            safety_state = "stable" if stability_margin > 0.03 else ("warning" if stability_margin > 0.0 else "unstable")
            power_elec = self.filtered_power_elec
            power_mech = self.filtered_power_mech
            symmetry_pct = accum_symmetry / substeps
            
            contact_forces = {k: v / substeps for k, v in accum_forces.items()}
            contacts = {k: (v / substeps) > 0.5 for k, v in accum_contacts.items()}
            stability_info = last_stability_info
            stability_info['stability_margin'] = stability_margin
            stability_info['state'] = safety_state
                
        # --- UPDATE GUI TELEMETRY WIDGETS ---
        self.telemetry_panel.update_telemetry(stability_margin, safety_state, power_elec, symmetry_pct)
        self.timeline_widget.update_states(contacts)
        self.plots_canvas.update_plots(np.array([0.0, 0.0]), stability_info, contact_forces, power_elec, power_mech, self.sim.dt)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Up:
            val = self.sliders["stride_x"].value() + 1
            self.sliders["stride_x"].setValue(int(np.clip(val, -15, 15)))
        elif event.key() == Qt.Key.Key_Down:
            val = self.sliders["stride_x"].value() - 1
            self.sliders["stride_x"].setValue(int(np.clip(val, -15, 15)))
        elif event.key() == Qt.Key.Key_Left:
            val = self.sliders["yaw_rate"].value() + 1
            self.sliders["yaw_rate"].setValue(int(np.clip(val, -5, 5)))
        elif event.key() == Qt.Key.Key_Right:
            val = self.sliders["yaw_rate"].value() - 1
            self.sliders["yaw_rate"].setValue(int(np.clip(val, -5, 5)))
        elif event.key() == Qt.Key.Key_PageUp:
            val = self.sliders["stride_y"].value() + 1
            self.sliders["stride_y"].setValue(int(np.clip(val, -8, 8)))
        elif event.key() == Qt.Key.Key_PageDown:
            val = self.sliders["stride_y"].value() - 1
            self.sliders["stride_y"].setValue(int(np.clip(val, -8, 8)))
        elif event.key() == Qt.Key.Key_Home:
            val = self.sliders["roll_offset"].value() - 2
            self.sliders["roll_offset"].setValue(int(np.clip(val, -30, 30)))
        elif event.key() == Qt.Key.Key_End:
            val = self.sliders["roll_offset"].value() + 2
            self.sliders["roll_offset"].setValue(int(np.clip(val, -30, 30)))
        elif event.key() == Qt.Key.Key_Insert:
            val = self.sliders["pitch_offset"].value() + 2
            self.sliders["pitch_offset"].setValue(int(np.clip(val, -30, 30)))
        elif event.key() == Qt.Key.Key_Delete:
            val = self.sliders["pitch_offset"].value() - 2
            self.sliders["pitch_offset"].setValue(int(np.clip(val, -30, 30)))
        elif event.key() == Qt.Key.Key_Space:
            self.sliders["stride_x"].setValue(0)
            self.sliders["stride_y"].setValue(0)
            self.sliders["yaw_rate"].setValue(0)
            self.sliders["roll_offset"].setValue(0)
            self.sliders["pitch_offset"].setValue(0)
            self.sliders["yaw_offset"].setValue(0)
            print("[*] E-Brake: Locomotion vectors zeroed.")
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.sim.close()
        event.accept()

    def mousePressEvent(self, event):
        self.setFocus()
        super().mousePressEvent(event)
