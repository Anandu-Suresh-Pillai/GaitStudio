from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QBrush
from PyQt6.QtCore import Qt, QTimer, QRectF
import numpy as np

class ScrollingGaitTimeline(QWidget):
    def __init__(self, parent=None, history_len=150):
        super().__init__(parent)
        self.history_len = history_len
        self.setMinimumHeight(150)
        
        # Color palette (Isaac Sim styled)
        self.color_bg = QColor(22, 23, 26)         # Deep slate
        self.color_stance = QColor(57, 255, 20)    # Neon Green
        self.color_swing = QColor(43, 44, 49)      # Charcoal dark
        self.color_grid = QColor(60, 60, 65)
        self.color_text = QColor(220, 220, 225)
        
        # Contact histories: 0=swing, 1=stance
        self.histories = {leg: [1]*self.history_len for leg in ['FL', 'FR', 'RL', 'RR']}
        self.legs = ['FL', 'FR', 'RL', 'RR']

    def update_states(self, current_contacts):
        """
        Appends the latest contact state and shifts history.
        current_contacts: dict {'FL': bool, ...}
        """
        for leg in self.legs:
            state = 1 if current_contacts[leg] else 0
            self.histories[leg].append(state)
            if len(self.histories[leg]) > self.history_len:
                self.histories[leg].pop(0)
        self.update() # trigger paintEvent

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # 1. Draw Background
        painter.fillRect(0, 0, width, height, self.color_bg)
        
        # Layout metrics
        margin_left = 60
        margin_top = 20
        margin_bottom = 15
        margin_right = 20
        
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom
        row_h = plot_h / 4
        
        # 2. Draw row backgrounds, lines, and text
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font)
        
        for idx, leg in enumerate(self.legs):
            y_top = margin_top + idx * row_h
            
            # Draw row label
            painter.setPen(QPen(self.color_text))
            painter.drawText(15, int(y_top + row_h/2 + 5), leg)
            
            # Draw row division line
            painter.setPen(QPen(self.color_grid, 1, Qt.PenStyle.DashLine))
            painter.drawLine(margin_left, int(y_top + row_h), width - margin_right, int(y_top + row_h))
            
            # Draw Gantt block rectangles
            hist = self.histories[leg]
            block_w = plot_w / self.history_len
            
            for t_idx, val in enumerate(hist):
                x_pos = margin_left + t_idx * block_w
                rect = QRectF(x_pos, y_top + 3, block_w + 0.5, row_h - 6)
                
                if val == 1:
                    painter.fillRect(rect, self.color_stance)
                else:
                    painter.fillRect(rect, self.color_swing)
                    
        # 3. Draw vertical playhead line indicating the "now" state
        painter.setPen(QPen(QColor(255, 50, 50), 2)) # active crimson line
        painter.drawLine(width - margin_right, margin_top, width - margin_right, height - margin_bottom)
        
        # Label playhead
        painter.setFont(QFont("Segoe UI", 7))
        painter.drawText(width - 45, margin_top - 5, "NOW")
        
        # Draw bounding border
        painter.setPen(QPen(self.color_grid, 1.5))
        painter.drawRect(margin_left, margin_top, plot_w, plot_h)


class TelemetryDashboardPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(15)
        
        # Status Card 1: Stability
        self.card_stability = TelemetryCard("STABILITY MARGIN", "0.00 m", "STABLE", QColor(57, 255, 20))
        self.layout.addWidget(self.card_stability)
        
        # Status Card 2: Power consumption
        self.card_power = TelemetryCard("SYSTEM POWER", "0.0 W", "STANDBY", QColor(0, 191, 255))
        self.layout.addWidget(self.card_power)
        
        # Status Card 3: Slip Ratio
        self.card_slip = TelemetryCard("GAIT SYMMETRY", "100.0 %", "OPTIMAL", QColor(57, 255, 20))
        self.layout.addWidget(self.card_slip)

    def update_telemetry(self, stability_margin, safety_state, power_w, symmetry_pct):
        # Update Stability card
        self.card_stability.value_label.setText(f"{stability_margin:.3f} m")
        self.card_stability.sub_label.setText(safety_state.upper())
        if safety_state == "stable":
            self.card_stability.set_accent(QColor(57, 255, 20)) # green
        elif safety_state == "warning":
            self.card_stability.set_accent(QColor(255, 176, 0)) # amber
        else:
            self.card_stability.set_accent(QColor(255, 51, 51)) # crimson
            
        # Update Power card
        self.card_power.value_label.setText(f"{power_w:.1f} W")
        if power_w < 12.0:
            self.card_power.sub_label.setText("STANDBY")
            self.card_power.set_accent(QColor(0, 191, 255)) # light blue
        else:
            self.card_power.sub_label.setText("DYNAMIC LOCOMOTION")
            self.card_power.set_accent(QColor(138, 43, 226)) # purple
            
        # Update Symmetry card
        self.card_slip.value_label.setText(f"{symmetry_pct:.1f} %")
        if symmetry_pct > 85.0:
            self.card_slip.sub_label.setText("OPTIMAL")
            self.card_slip.set_accent(QColor(57, 255, 20))
        elif symmetry_pct > 65.0:
            self.card_slip.sub_label.setText("MODERATE ASYMMETRY")
            self.card_slip.set_accent(QColor(255, 176, 0))
        else:
            self.card_slip.sub_label.setText("SEVERE GAIT ASYMMETRY")
            self.card_slip.set_accent(QColor(255, 51, 51))


class TelemetryCard(QWidget):
    def __init__(self, title, value, subtext, accent_color, parent=None):
        super().__init__(parent)
        self.accent_color = accent_color
        
        # Design layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 12, 15, 12)
        self.layout.setSpacing(6)
        
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: #7a7c85; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        self.layout.addWidget(self.title_label)
        
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("color: #ffffff; font-size: 24px; font-weight: bold; font-family: Consolas;")
        self.layout.addWidget(self.value_label)
        
        self.sub_label = QLabel(subtext)
        self.sub_label.setStyleSheet(f"color: {accent_color.name()}; font-size: 10px; font-weight: bold;")
        self.layout.addWidget(self.sub_label)
        
        self.setStyleSheet("background-color: #1a1b1f; border-radius: 8px; border: 1px solid #2f3036;")

    def set_accent(self, color):
        self.accent_color = color
        self.sub_label.setStyleSheet(f"color: {color.name()}; font-size: 10px; font-weight: bold;")
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        # Draw a beautiful accent left border block
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(self.accent_color))
        painter.setPen(Qt.PenStyle.NoPen)
        # Draw a 4px wide bar on the left edge
        painter.drawRoundedRect(QRectF(0, 0, 4, self.height()), 2, 2)
