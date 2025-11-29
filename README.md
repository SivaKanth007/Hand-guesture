# Hand Gesture Mouse Controller 🖱️👋

A Python-based AI application that allows you to control your mouse cursor using hand gestures captured by your webcam. This prototype serves as a foundation for touchless interfaces and hologram-style interactions.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand_Tracking-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green)

## 🌟 Features

*   **Touchless Navigation**: Move the mouse cursor by simply moving your index finger.
*   **Gesture Clicking**: Pinch your index finger and thumb to click or drag items.
*   **Real-time Tracking**: Uses MediaPipe for low-latency, high-precision hand tracking.
*   **Smooth Motion**: Implements exponential smoothing to reduce cursor jitter.
*   **Non-Blocking UI**: Multi-threaded architecture ensures smooth mouse control even when moving application windows.
*   **Safety First**: Automatic mouse release when tracking is lost or the app exits.

## 🛠️ Prerequisites

*   **Hardware**: A webcam connected to your computer.
*   **OS**: Windows, macOS, or Linux.
*   **Python**: Version 3.8 or higher.

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/SivaKanth007/hand-gesture-controller.git
    cd hand-gesture-controller
    ```

2.  **Post-clone setup (virtual environment or system Python)**

    After cloning this repository you have two common options to run the project:

    Option A — Create and use a local virtual environment (recommended):

    ```powershell
    # Create venv
    python -m venv venv

    # PowerShell (activate)
    .\venv\Scripts\Activate.ps1

    # cmd.exe (activate)
    .\venv\Scripts\activate.bat

    # macOS / Linux (bash/zsh)
    python3 -m venv venv
    source venv/bin/activate
    ```

    Once the virtual environment is active, install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    Note: This project does not include a `venv` directory — do not commit your local virtual environment. The `.gitignore` already excludes `venv/`.

    Option B — Use your system Python (no virtual environment):

    If you prefer not to create a virtual environment, install the dependencies directly using your system Python. This will install packages globally or into your user site-packages depending on your pip configuration.

    ```bash
    pip install -r requirements.txt
    # or to install for the current user without admin rights:
    pip install --user -r requirements.txt
    ```

    Choose the option that fits your workflow. Using a virtual environment is recommended to avoid dependency conflicts with other Python projects on your machine.

## 🚀 Usage

1.  **Run the application**
    ```bash
    python main.py
    ```

2.  **Controls**
    *   **Move Cursor**: Point with your **Index Finger**. The cursor follows your **Knuckle** for maximum stability.
    *   **Left Click**: Pinch **Index Finger** and **Thumb** quickly.
    *   **Double Click**: Pinch **Index Finger** and **Thumb** twice quickly.
    *   **Right Click**: Pinch **Middle Finger** and **Thumb**.
    *   **Drag & Drop**: Pinch **Index Finger** and **Thumb** and **HOLD** for 0.5 seconds. The cursor will unlock and follow your hand.
    *   **Minimize Windows**: Make a **Fist** (0 fingers up).
    *   **Restore Windows**: Show an **Open Palm** (5 fingers up).
    *   **Exit**: Press `Esc` to close.

3.  **Features**
    *   **FPS Counter**: Real-time performance monitoring in the top-left corner.
    *   **Anti-Jitter**: Uses "One Euro Filter" and a Deadzone for buttery smooth precision.
    *   **Smart Locking**: Cursor freezes during clicks to prevent accidental movement.

## ⚙️ Configuration

You can tweak the sensitivity in `main.py`:

*   **Smoothing**: Adjusted via `OneEuroFilter` parameters in `HandController.__init__`.
*   **Deadzone**: `self.deadzone = 3` (pixels).
*   **Drag Delay**: `0.5` seconds (in `process_gestures`).

## 🔧 Troubleshooting

### Common Issues

*   **"Camera not found" or Black Screen**:
    *   Ensure your webcam is connected and not being used by another application (like Zoom or Teams).
    *   If you have multiple cameras, you might need to change the camera index in `main.py`. Look for `cap = cv2.VideoCapture(0)` and change `0` to `1` or `2`.
    *   Check if you have proper drivers installed for your webcam.

*   **Cursor is jittery or unstable**:
    *   **Lighting**: Ensure your hand is well-lit. Shadows can confuse the tracking.
    *   **Background**: A busy background might interfere. Try a plain wall.
    *   **Smoothing**: Increase the `smoothing_factor` in `main.py` (e.g., to `0.1`) for smoother but slower movement.

*   **Clicks are not registering or getting stuck**:
    *   **Hand Orientation**: Ensure your hand is facing the camera directly.
    *   **Distance**: You may be too far or too close. Try moving your hand to a comfortable distance (approx. 0.5 - 1 meter).
    *   **Thresholds**: Adjust `click_start_threshold` (default `0.05`) in `main.py` if you have larger or smaller hands.

*   **Permission Errors (macOS/Linux)**:
    *   If the mouse doesn't move, you likely need to grant "Accessibility" or "Input Monitoring" permissions to your Terminal or IDE in your System Preferences.

*   **Installation Errors**:
    *   If `pip install` fails, try upgrading pip: `pip install --upgrade pip`.
    *   Ensure you are using Python 3.8 - 3.11 (MediaPipe sometimes has delays supporting the absolute newest Python versions).

### Emergency Stop
*   The application disables the default PyAutoGUI failsafe (slamming mouse to corner) to prevent accidental triggers during gestures.
*   **To Exit**: Press the **`Esc`** key on your keyboard.
*   **Force Quit**: If the app freezes, switch to the terminal window (Alt+Tab) and press `Ctrl+C`.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
