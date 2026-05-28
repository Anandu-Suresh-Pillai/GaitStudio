import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class LivePlotsCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=3.5, dpi=100):
        # Premium Dark theme setup for Matplotlib
        self.color_bg = "#16171a"         # matches QColor(22, 23, 26)
        self.color_panel = "#1a1b1f"      # matches QColor(26, 27, 31)
        self.color_grid = "#2f3036"
        self.color_text = "#dcdee5"
        self.color_stance = "#39ff14"     # neon green
        self.color_swing = "#ff3333"      # red
        
        # Initialize Figure
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor=self.color_bg)
        super().__init__(fig)
        self.setParent(parent)
        
        # Create subplots side-by-side
        # 1. Left: Stability & Support Polygon (2D scatter & polygon)
        # 2. Middle: Contact Forces (4-bar chart)
        # 3. Right: Power Consumption (rolling curves)
        self.ax_stability = fig.add_subplot(131, facecolor=self.color_panel)
        self.ax_forces = fig.add_subplot(132, facecolor=self.color_panel)
        self.ax_power = fig.add_subplot(133, facecolor=self.color_panel)
        
        fig.tight_layout(pad=3.0)
        
        # Power history buffer
        self.history_len = 100
        self.time_history = list(np.linspace(-5, 0, self.history_len))
        self.power_elec_history = [8.0] * self.history_len
        self.power_mech_history = [0.0] * self.history_len
        
        # Set titles and styling
        self.setup_stability_axis()
        self.setup_forces_axis()
        self.setup_power_axis()

    def setup_stability_axis(self):
        ax = self.ax_stability
        ax.set_title("SUPPORT POLYGON & COM", color=self.color_text, fontsize=9, fontweight='bold', pad=10)
        ax.set_xlim(-0.35, 0.35)
        ax.set_ylim(-0.25, 0.25)
        ax.set_xlabel("X (Forward) [m]", color=self.color_text, fontsize=8)
        ax.set_ylabel("Y (Lateral) [m]", color=self.color_text, fontsize=8)
        ax.grid(True, color=self.color_grid, linestyle=':')
        ax.tick_params(colors=self.color_text, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(self.color_grid)
            
        # Draw a default base footprint representing the hips
        ax.plot([0.185, 0.185, -0.185, -0.185, 0.185], [0.10, -0.10, -0.10, 0.10, 0.10], 
                color="#50525b", linestyle="--", linewidth=1.2, label="Torso Hips")

    def setup_forces_axis(self):
        ax = self.ax_forces
        ax.set_title("FOOT CONTACT FORCES", color=self.color_text, fontsize=9, fontweight='bold', pad=10)
        ax.set_ylim(0, 50)
        ax.set_ylabel("Normal Force [N]", color=self.color_text, fontsize=8)
        ax.grid(True, axis='y', color=self.color_grid, linestyle=':')
        ax.tick_params(colors=self.color_text, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(self.color_grid)
            
        self.bar_labels = ['FL', 'FR', 'RL', 'RR']
        self.bars = ax.bar(self.bar_labels, [0, 0, 0, 0], color='#8a2be2', edgecolor=self.color_grid, width=0.5)

    def setup_power_axis(self):
        ax = self.ax_power
        ax.set_title("POWER CONSUMPTION", color=self.color_text, fontsize=9, fontweight='bold', pad=10)
        ax.set_xlim(-5, 0)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Time History [s]", color=self.color_text, fontsize=8)
        ax.set_ylabel("Power [W]", color=self.color_text, fontsize=8)
        ax.grid(True, color=self.color_grid, linestyle=':')
        ax.tick_params(colors=self.color_text, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(self.color_grid)
            
        self.line_elec, = ax.plot(self.time_history, self.power_elec_history, color='#00bfff', linewidth=1.5, label="Electrical")
        self.line_mech, = ax.plot(self.time_history, self.power_mech_history, color='#8a2be2', linewidth=1.5, label="Mechanical")
        ax.legend(facecolor=self.color_panel, edgecolor=self.color_grid, labelcolor=self.color_text, fontsize=7, loc='upper left')

    def update_plots(self, com_xy, stability_info, contact_forces, power_elec, power_mech, dt):
        """
        Refreshes all plots with the latest telemetry values.
        """
        # 1. Update Stability Plot
        ax_s = self.ax_stability
        
        # Remove old dynamic plots (keep only the torso box outline)
        for line in list(ax_s.lines):
            if line.get_label() != "Torso Hips":
                line.remove()
        for poly in list(ax_s.collections):
            poly.remove()
        for patch in list(ax_s.patches):
            patch.remove()
            
        # Draw support polygon CCW convex hull
        poly_pts = stability_info['support_polygon']
        if len(poly_pts) >= 3:
            # Sort for plotting matching standard CCW loop
            x_pts = [p[0] for p in poly_pts] + [poly_pts[0][0]]
            y_pts = [p[1] for p in poly_pts] + [poly_pts[0][1]]
            
            # Draw fill
            state = stability_info['state']
            f_color = self.color_stance if state == 'stable' else ('#ffb000' if state == 'warning' else self.color_swing)
            ax_s.fill(x_pts, y_pts, facecolor=f_color, alpha=0.15, label="Support Poly")
            ax_s.plot(x_pts, y_pts, color=f_color, linewidth=2, linestyle='-', marker='o', markersize=4)
        elif len(poly_pts) == 2:
            # Draw line segment
            x_pts = [p[0] for p in poly_pts]
            y_pts = [p[1] for p in poly_pts]
            ax_s.plot(x_pts, y_pts, color='#ffb000', linewidth=2, marker='o', markersize=4)
            
        # Draw COM projection crosshair (+)
        com_color = self.color_stance if stability_info['state'] == 'stable' else self.color_swing
        ax_s.plot([com_xy[0]], [com_xy[1]], marker='P', color=com_color, markersize=8, markeredgecolor='black', label="COM")
        
        # Draw projection line to closest edge if stable and has closest point
        if 'closest_point' in stability_info and stability_info['closest_point'] is not None:
            c_pt = stability_info['closest_point']
            ax_s.plot([com_xy[0], c_pt[0]], [com_xy[1], c_pt[1]], color='#00bfff', linestyle=':', linewidth=1.5)
            
        # 2. Update Contact Forces Bar Chart
        for bar, leg in zip(self.bars, self.bar_labels):
            force = contact_forces[leg]
            bar.set_height(force)
            # Paint dynamic colors based on load
            if force > 35:
                bar.set_color(self.color_swing) # high impact
            elif force > 5:
                bar.set_color('#8a2be2') # standard load
            else:
                bar.set_color(self.color_grid) # stance exit/swing
                
        # 3. Update Power curves
        self.power_elec_history.append(power_elec)
        self.power_mech_history.append(power_mech)
        
        self.power_elec_history.pop(0)
        self.power_mech_history.pop(0)
        
        self.line_elec.set_ydata(self.power_elec_history)
        self.line_mech.set_ydata(self.power_mech_history)
        
        # Scale power axis limits dynamically if power spikes
        max_p = max(max(self.power_elec_history), 50.0)
        self.ax_power.set_ylim(0, max_p * 1.15)
        
        # Redraw
        self.draw_idle()
