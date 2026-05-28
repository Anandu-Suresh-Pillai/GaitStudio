import sys
from PyQt6.QtWidgets import QApplication
from gui.dashboard import GaitStudioDashboard

def main():
    print("==================================================")
    print("       STARTING GAITSTUDIO CONTROL DASHBOARD      ")
    print("==================================================")
    
    # Initialize PyQt application
    app = QApplication(sys.argv)
    
    # Create and display the main dashboard window
    dashboard = GaitStudioDashboard()
    dashboard.show()
    
    # Run the application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
