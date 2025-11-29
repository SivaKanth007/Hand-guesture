import numpy as np
import math

def map_coordinates(x, y, src_w, src_h, dst_w, dst_h):
    """
    Maps coordinates from source resolution (webcam) to destination resolution (screen).
    Includes a margin to allow reaching edges easily.
    """
    # Define a margin to make it easier to reach the edges of the screen
    margin = 100  # pixels
    
    # Clamp x and y to the effective area
    x_clamped = np.clip(x * src_w, margin, src_w - margin)
    y_clamped = np.clip(y * src_h, margin, src_h - margin)
    
    # Normalize back to 0-1 range within the margin
    x_norm = (x_clamped - margin) / (src_w - 2 * margin)
    y_norm = (y_clamped - margin) / (src_h - 2 * margin)
    
    # Map to destination
    screen_x = np.interp(x_norm, [0, 1], [0, dst_w])
    screen_y = np.interp(y_norm, [0, 1], [0, dst_h])
    
    return int(screen_x), int(screen_y)

def calculate_distance(p1, p2):
    """
    Calculates Euclidean distance between two MediaPipe landmarks.
    """
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        Initialize the One Euro Filter.
        min_cutoff: Min cutoff frequency in Hz. Lower = more smoothing at low speeds.
        beta: Speed coefficient. Higher = less lag at high speeds.
        d_cutoff: Cutoff frequency for derivative.
        """
        self.t_prev = t0
        self.x_prev = x0
        self.dx_prev = dx0
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        """
        Filter the signal.
        t: Current timestamp.
        x: Current value.
        """
        t_e = t - self.t_prev
        
        # Prevent division by zero or negative time
        if t_e <= 0:
            return self.x_prev

        # Estimate derivative (speed)
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # Calculate cutoff based on speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        
        # Filter the signal
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat
