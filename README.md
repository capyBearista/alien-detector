# 👽 Alien Hand Scanner 9000
*Halloween Project @ Quincy College Computer Club by **🧟‍♀️ Thanh van Nguyen** and **🎃 Arjun Pramanik***

Detect alien hands using 100% effective science! This project uses a webcam to analyze hand landmarks and classify hands as "Alien" or "Human" based on finger proportions. Capture and save detected alien hands, and play themed videos and sounds for each detection.

## Features
- Real-time hand landmark detection using MediaPipe
- Classifies hands as "Alien" or "Human" based on finger length difference
- Saves images of detected alien hands
- Plays themed video/audio on detection
- Palm line detection test script (see `palm.py`)
- Works with webcam or test images

## Demo [TBD]
![](demo-images/hand1.png)
![](demo-images/hand2.png)
![](demo-images/hand3.png)

## Getting Started
### Prerequisites
- **Python 3.12** or before
- Webcam

### Installation
1. Clone this repo
	```powershell
	git clone <LINK_HERE>
	cd alien-detector
	```
2. Set up the virtual environment and activate it
	```powershell
	pip install virtualenv
	virtualenv .venv
    .venv\Scripts\Activate
	```
3. Install dependencies:
	```powershell
	pip install -r requirements.txt
	```

<!-- 4. Download the MediaPipe hand landmark model (`hand_landmarker.task`) and place it in the project folder.
   - https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker -->

### Usage
Run the main detector (ensure the virtual environment is activated first):
```bash
python main.py
```

Controls:
- `c`: Capture and classify current hand
- `q`: Quit

> [!WARNING]
> By using this script, you acknowledge that you may be assisting a top-secret government operation in the detection of extraterrestrial hand activity. The authors accept no responsibility for the results of this program. Use at your own risk, alien!

## Project Structure
- `main.py`: Main alien hand detector
- `palm.py`: Palm line detection (test script)
- `hand_landmarker.task`: Pre-trained machine learning model to identify hands and hand landmarks, by Google, for use with MediaPipe
- `alien_hands_detected/`: Saved detection result images
- `vids/`: Saved video clips
- `demo-images/`: Demo images showcasing the model at work
- `result-vids/`: Media files for detection events
- `other/`: Contains Blender animation by Thanh for other Halloween project work

## Additional Notes
This project was developed on Windows 11 and tested with PowerShell. WSL may have issues accessing cameras, so running on Linux could require additional setup.

## Authors
- Thanh van Nguyen
- Arjun Pramanik

## License
[BSD 3-Clause](LICENSE.md)