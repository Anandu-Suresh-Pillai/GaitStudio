import csv
import os

class LocomotionReplayer:
    def __init__(self):
        self.keyframes = []
        self.current_idx = 0
        self.is_loaded = False

    def load_session(self, filepath):
        """
        Loads a CSV session recording.
        """
        if not os.path.exists(filepath):
            print(f"[-] Session file not found: {filepath}")
            return False
            
        self.keyframes = []
        try:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert all fields to float
                    parsed_row = {}
                    for k, v in row.items():
                        try:
                            parsed_row[k] = float(v)
                        except ValueError:
                            parsed_row[k] = v # keep string/bool
                    self.keyframes.append(parsed_row)
            
            self.current_idx = 0
            self.is_loaded = len(self.keyframes) > 0
            print(f"[+] Loaded {len(self.keyframes)} keyframes for playback from {filepath}")
            return self.is_loaded
        except Exception as e:
            print(f"[-] Error loading playback session: {e}")
            return False

    def get_next_frame(self):
        """
        Returns the next frame in the sequence, wrapping around or returning None if done.
        """
        if not self.is_loaded or not self.keyframes:
            return None
            
        frame = self.keyframes[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.keyframes)
        return frame

    def reset_playback(self):
        self.current_idx = 0
        
    def get_num_frames(self):
        return len(self.keyframes)
