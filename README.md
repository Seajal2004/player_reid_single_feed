# Player Re-identification Assignment - Option 2

Solution for player re-identification in a single video feed, ensuring players maintain consistent IDs when going out of frame and reappearing.

## Required Files Setup

### 1. Download Model

Download the YOLOv11 model from:

```
https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view
```

- Save as `models/best.pt`

### 2. Download Video

Download the input video from:

```
https://drive.google.com/file/d/1TDcND31fvEDvcnZCaianTxJrmT8q7iIi/view?usp=sharing
```

- Save `15sec_input_720p.mp4` in `videos/` folder

## Installation & Setup

1. **Clone repository**

```bash
git clone <https://github.com/Seajal2004/player_reid_single_feed.git>
cd player_reid_single_feed
```

2. **Create virtual environment**

```bash
python -m venv venv
```

3. **Activate virtual environment**

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

5. **Create directories**

```bash
mkdir -p models videos output
```

6. **Place downloaded files**

- Put `best.pt` in `models/` folder
- Put `15sec_input_720p.mp4` in `videos/` folder

## Run Solution

```bash
python main.py
```

## Output

Tracked video will be saved as `output/tracked_video.mp4`

## Solution Features

- ✅ Consistent player IDs throughout video
- ✅ Re-identification when players reappear
- ✅ Handles occlusions and temporary disappearances
- ✅ Real-time processing capability

## File Structure

```
player_reid_single_feed/
├── models/best.pt              # Download required
├── videos/15sec_input_720p.mp4 # Download required
├── output/tracked_video.mp4    # Generated output
├── src/
│   ├── detector.py            # Player detection
│   └── stable_tracker.py      # ID tracking
├── main.py                    # Main script
├── requirements.txt           # Dependencies
└── README.md                  # This file
```
"# player_reid_single_feed" 
