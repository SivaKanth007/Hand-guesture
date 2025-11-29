import cv2
import mediapipe as mp
import pyautogui
import time
import threading
from utils import map_coordinates, calculate_distance, OneEuroFilter

# Global variables for thread communication
frame_lock = threading.Lock()
current_frame = None
running = True

class HandController:
    def __init__(self):
        # Screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()
        
        # PyAutoGUI Optimization
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = False
        
        # One Euro Filters for X and Y
        t0 = time.time()
        # min_cutoff: 0.01 (Very heavy smoothing when slow), beta: 0.05 (Low speed sensitivity for "heavy" feel)
        self.filter_x = OneEuroFilter(t0, 0, min_cutoff=0.01, beta=0.05)
        self.filter_y = OneEuroFilter(t0, 0, min_cutoff=0.01, beta=0.05)
        
        # State
        self.left_clicked = False
        self.right_clicked = False
        
        # Cursor Locking (to prevent drift during pinch)
        self.cursor_locked = False
        self.locked_x = 0
        self.locked_y = 0
        
        # Thresholds
        self.click_start_thresh = 0.05
        self.click_stop_thresh = 0.06
        self.lock_thresh = 0.1 # Increased for Double Click stability
        self.deadzone = 3 # Pixels
        
        # Drag State
        self.pinch_start_time = 0
        self.is_pinching = False
        
        # Window State
        self.window_minimized = False
        self.last_window_toggle = 0
        
    def count_fingers(self, landmarks):
        """
        Counts the number of fingers up.
        Returns: int (0-5)
        """
        mp_hands = mp.solutions.hands
        fingers = []
        
        # Thumb (Check x-coordinate relative to MCP for left/right hand logic, 
        # but for simplicity we check if tip is to the left/right of IP joint depending on hand side.
        # Assuming Right Hand for now or general vertical check if hand is upright)
        # Simple heuristic: Check if tip is higher (lower y) than IP joint? No, thumb moves sideways.
        # Robust heuristic: Compare Tip to IP joint distance from MCP.
        
        # Simplified Finger Counting (Tip y < PIP y)
        # Index
        if landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
            fingers.append(1)
        # Middle
        if landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y:
            fingers.append(1)
        # Ring
        if landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y:
            fingers.append(1)
        # Pinky
        if landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y:
            fingers.append(1)
            
        # Thumb (Tip x vs IP x is tricky depending on hand orientation. 
        # Let's use a simple check: is tip far from palm center?)
        # For this specific "Fist vs Open" check, checking the 4 fingers is usually enough.
        # If 4 fingers are down, it's likely a fist. If 4 are up, it's open.
        
        return len(fingers)

    def process_gestures(self, landmarks, image_shape):
        h, w, _ = image_shape
        mp_hands = mp.solutions.hands
        
        # Get key landmarks
        index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_mcp = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        # --- 0. Window Management (Fist / Open Palm) ---
        fingers_up = self.count_fingers(landmarks)
        curr_time = time.time()
        
        # Debounce toggle (1 second cooldown)
        if curr_time - self.last_window_toggle > 1.0:
            if fingers_up == 0: # Fist
                if not self.window_minimized:
                    pyautogui.hotkey('win', 'd')
                    self.window_minimized = True
                    self.last_window_toggle = curr_time
                    return False, False, (0,0) # Stop processing mouse
            elif fingers_up == 4: # Open Palm (Thumb logic excluded for simplicity, so 4 fingers)
                if self.window_minimized:
                    pyautogui.hotkey('win', 'd')
                    self.window_minimized = False
                    self.last_window_toggle = curr_time
                    return False, False, (0,0)

        # --- 1. Cursor Movement (Using Knuckle/MCP) ---
        target_x, target_y = map_coordinates(
            index_mcp.x, index_mcp.y, w, h, self.screen_w, self.screen_h
        )
        
        smooth_x = self.filter_x(curr_time, target_x)
        smooth_y = self.filter_y(curr_time, target_y)
        
        # --- 2. Dynamic Cursor Locking & Deadzone ---
        dist_left = calculate_distance(index_tip, thumb_tip)
        
        # Logic: Lock cursor initially, but UNLOCK if held for dragging
        if dist_left < self.lock_thresh:
            if not self.is_pinching:
                self.is_pinching = True
                self.pinch_start_time = curr_time
                # Lock initially
                if not self.cursor_locked:
                    self.cursor_locked = True
                    self.locked_x = smooth_x
                    self.locked_y = smooth_y
            
            # Check duration
            pinch_duration = curr_time - self.pinch_start_time
            if pinch_duration > 0.5: # 0.5s threshold for Drag Mode
                self.cursor_locked = False # Unlock for drag
                
            if self.cursor_locked:
                final_x, final_y = self.locked_x, self.locked_y
            else:
                final_x, final_y = smooth_x, smooth_y # Follow hand for drag
        else:
            self.is_pinching = False
            self.cursor_locked = False
            
            # Deadzone Logic
            curr_mouse_x, curr_mouse_y = pyautogui.position()
            if abs(smooth_x - curr_mouse_x) > self.deadzone or abs(smooth_y - curr_mouse_y) > self.deadzone:
                final_x, final_y = smooth_x, smooth_y
            else:
                final_x, final_y = curr_mouse_x, curr_mouse_y
            
        # Move mouse
        pyautogui.moveTo(final_x, final_y)
        
        # --- 3. Left Click (Index + Thumb) ---
        if dist_left < self.click_start_thresh:
            if not self.left_clicked:
                pyautogui.mouseDown()
                self.left_clicked = True
        elif dist_left > self.click_stop_thresh:
            if self.left_clicked:
                pyautogui.mouseUp()
                self.left_clicked = False
                
        # --- 4. Right Click (Middle + Thumb) ---
        dist_right = calculate_distance(middle_tip, thumb_tip)
        if dist_right < self.click_start_thresh:
            if not self.right_clicked:
                pyautogui.rightClick()
                self.right_clicked = True
        elif dist_right > self.click_stop_thresh:
            self.right_clicked = False
            
        return self.left_clicked, self.right_clicked, (int(index_tip.x * w), int(index_tip.y * h))

def gesture_processing_thread():
    global current_frame, running
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize Controller
    controller = HandController()
    
    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    
    # Optimize Camera (Try to force 60 FPS)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # FPS Calculation
    prev_time = 0
    
    with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=1, # 0=Lite, 1=Full (Better accuracy)
        max_num_hands=1
    ) as hands:
        
        while running and cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            # Flip and convert
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)

            # Draw annotations
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            
            # Display FPS
            cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Process Gestures
                    l_click, r_click, cursor_pos = controller.process_gestures(
                        hand_landmarks, image.shape
                    )
                    
                    # Visual Feedback
                    if l_click:
                        cv2.circle(image, cursor_pos, 15, (0, 255, 0), cv2.FILLED) # Green for Left
                    if r_click:
                        cv2.circle(image, cursor_pos, 15, (0, 0, 255), cv2.FILLED) # Red for Right
            
            # Update global frame safely
            with frame_lock:
                current_frame = image.copy()
                
    # Cleanup
    cap.release()

def main():
    global running
    print("Hand Gesture Controller Started. Press 'Esc' to exit.")
    
    # Start processing thread
    t = threading.Thread(target=gesture_processing_thread)
    t.daemon = True
    t.start()
    
    # Main GUI loop
    while True:
        with frame_lock:
            if current_frame is not None:
                cv2.imshow('Hand Gesture Controller', current_frame)
        
        # Wait for Esc key
        if cv2.waitKey(5) & 0xFF == 27:
            running = False
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
