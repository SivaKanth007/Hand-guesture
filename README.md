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
    *   **Move Cursor**: Raise your hand and point with your **Index Finger**. The cursor will follow your finger tip.
    *   **Left Click / Drag**: Pinch your **Index Finger** and **Thumb** together.
        *   *Quick Pinch*: Single Click.
        *   *Pinch & Hold*: Click and Drag.
    *   **Exit**: Press the `Esc` key to close the application.

## ⚙️ Configuration

You can tweak the sensitivity and behavior in `main.py`:

*   **Smoothing**: Adjust `smoothing_factor` (default `0.2`).
    *   Higher value (e.g., `0.5`) = Faster response, more jitter.
    *   Lower value (e.g., `0.1`) = Smoother movement, slightly more lag.
*   **Click Sensitivity**: Adjust `click_start_threshold` and `click_stop_threshold`.

## 🔧 Troubleshooting

*   **Cursor is jittery**: Try lowering the `smoothing_factor` or ensure your environment is well-lit.
*   **Clicks are not registering**: Ensure your hand is facing the camera clearly. You may need to adjust the `click_start_threshold` if your hand is very far or very close to the camera.
*   **Permissions**: Ensure your terminal/IDE has permission to access the webcam and control the mouse (especially on macOS).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
