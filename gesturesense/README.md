
# GestureSense

Real-time hand gesture recognition for the desktop, built on **MediaPipe Hands** and **OpenCV**. Tracks one or two hands from a webcam, recognises 14 gestures with confidence scoring and temporal stabilisation, and renders a clean HUD at 30–60 FPS.

```
┌────────────────────────────────────────────────────────────┐
│ GestureSense   58.3 FPS  cam 30.0 · inf 11.2 ms   TRACKING │
│ Right: Thumbs Up  ▓▓▓▓▓▓▓░░  Left: Peace / Victory ▓▓▓▓▓░░ │
│                                                            │
│                   [ live camera feed with                  │
│                     skeleton, bounding box                 │
│                     and gesture labels ]                   │
│                                                            │
│ NORMAL · MIRROR · DARK      Q quit F fullscreen D debug …  │
└────────────────────────────────────────────────────────────┘
```

## Supported gestures

Thumbs Up · Thumbs Down · Open Palm · Closed Fist · Peace / Victory · OK Sign · Pointing Up / Down / Left / Right · Rock Sign · Call Me (Shaka) · Love Sign (ILY) · Finger Gun

> **Note:** *Peace* and *Victory* are the same physical hand shape (index + middle in a V), so they are deliberately implemented as one rule with a combined label. Two identical rules would only produce a coin-flip between them.

## Installation

Requires **Python 3.10+**.

```bash
git clone <your-repo-url> gesturesense
cd gesturesense
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Linux note:** OpenCV's HighGUI needs a display server and the usual GUI libs. On a minimal install: `sudo apt install libgl1 libglib2.0-0`. If your webcam is not at `/dev/video0`, pass `--camera 1` (etc.), and make sure your user is in the `video` group.

**MediaPipe backends:** recent MediaPipe releases (≥ 0.10.15) removed the legacy `mp.solutions` API. GestureSense supports both generations automatically — the legacy Solutions backend when your installed wheel still ships it, and the current Tasks `HandLandmarker` otherwise. The Tasks backend needs a model file at `models/hand_landmarker.task`; a verified copy of Google's official model (Apache 2.0, SHA256 `fbc2a300…`) is bundled with this project, and it is re-downloaded automatically if ever missing.

## Running

```bash
python app.py                          # defaults + gesturesense/config/default_config.yaml
python app.py --config my.yaml         # custom configuration
python app.py --camera 1 --debug       # CLI overrides
python app.py --fullscreen
```

### Keyboard controls

| Key | Action |
| --- | ------ |
| `Q` / `Esc` | Quit |
| `F` | Toggle fullscreen |
| `D` | Toggle debug overlay (finger states, joint angles, timings, CPU/RAM) |
| `M` | Toggle mirror mode |
| `T` | Cycle theme (dark / light) |
| `S` | Save a screenshot to `logs/captures/` |

## Configuration

Everything tunable lives in `gesturesense/config/default_config.yaml` — camera resolution and FPS, MediaPipe confidences and model complexity, gesture thresholds and stabilisation window, theme, HUD toggles, logging. Copy it, edit what you need, and pass it with `--config`. Unknown keys are reported (not silently ignored), and out-of-range values fail fast at startup with a clear message.

## Architecture

```
app.py                      thin CLI entry point
gesturesense/
  camera/capture.py         threaded capture, auto-reconnect, mirror at source
  vision/hand_tracker.py    MediaPipe backend behind a swappable ABC
  tracking/smoother.py      per-hand EMA landmark smoothing + teleport reset
  gesture/
    topology.py             21-point hand model (no MediaPipe dependency)
    finger_state.py         soft per-finger extension scores
    features.py             landmarks → HandFeatures (the rules' only input)
    base.py                 GestureRule ABC + registry decorator
    rules/                  one small class per gesture
    engine.py               best-rule selection + temporal majority vote
  core/
    pipeline.py             inference worker thread (latest-value slots)
    application.py          wiring, render loop, keys, graceful shutdown
  ui/                       theme, hand renderer, HUD, window wrapper
  config/settings.py        typed dataclasses + YAML overrides + validation
  utils/                    geometry, timing, perf, logging helpers
tests/                      67 unit tests incl. synthetic-pose gesture tests
```

Three threads, deliberately queue-free: the **capture thread** and the **inference thread** each publish into a latest-value slot, and the **main-thread render loop** (a HighGUI requirement) always consumes the newest data. Stale frames are dropped instead of queued, which caps latency instead of letting it accumulate.

The gesture pipeline is `landmarks → feature extraction → finger states → rule engine → confidence → temporal stabilisation`. Every constraint is a smooth 0–1 score (Hermite ramps, not booleans), a rule's score is dominated by its weakest constraints, and a majority vote over the last 7 frames removes flicker before a gesture is announced.

## Adding a new gesture

No existing code changes — drop a new rule class in `gesturesense/gesture/rules/`:

```python
from gesturesense.gesture.base import GestureRule, register_gesture
from gesturesense.gesture.rules import scoring as sc

@register_gesture
class ThreeFingers(GestureRule):
    """Index, middle and ring extended."""
    name = "Three"

    def score(self, features) -> float:
        return self.all_of(
            sc.extended(features, "index"),
            sc.extended(features, "middle"),
            sc.extended(features, "ring"),
            sc.curled(features, "pinky"),
            sc.not_extended(features, "thumb"),
        )
```

Import it from `gesturesense/gesture/rules/__init__.py`, add a synthetic pose in `tests/fixtures.py`, and the parametrised test suite covers it automatically. An ML classifier can plug in the same way: implement `score()` on top of `HandFeatures` (or raw `features.landmarks`) and register it.

## Testing

```bash
python -m pytest tests/ -v
```

Gesture logic is fully decoupled from MediaPipe, so the entire recognition pipeline is tested against parametric synthetic hands — no camera required.

## Troubleshooting

- **"Camera failed" banner** — device busy or wrong index; try `--camera 1`, close other apps using the webcam, check OS camera permissions.
- **Low FPS** — lower `camera.width/height` in the config, or set `tracking.model_complexity: 0` (faster, slightly less accurate).
- **Gestures flicker** — raise `gestures.history` / `gestures.min_votes` for more stability (at the cost of a few frames of latency), or improve lighting.
- **Left/right labels swapped** — press `M`; handedness labels track the mirror state automatically.
- **`ImportError: libGL.so.1` (Linux)** — `sudo apt install libgl1`.

## Roadmap hooks

The architecture is ready for gesture-driven automation: a future `actions/` module can subscribe to `RecognizedGesture` events from the pipeline output without touching recognition code — media keys, presentation control, volume/brightness, macros, or a custom-gesture trainer feeding new `GestureRule` implementations.
