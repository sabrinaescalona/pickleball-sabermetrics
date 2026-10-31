"""Conservative motion-based rally candidate detection. Not ball/shot detection."""
from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path


def analyze_video(path: str, sample_fps: float = 4.0, min_rally_seconds: float = 3.0):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('Could not open video. Try an MP4 encoded with H.264.')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frames <= 0:
        cap.release()
        raise ValueError('Video has invalid duration or frame rate.')
    duration = frames / fps
    stride = max(1, round(fps / sample_fps))
    samples = []
    previous = None
    for idx in range(0, frames, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        # Central court region avoids much of the spectator/scoreboard motion.
        crop = frame[int(h*.13):int(h*.92), int(w*.12):int(w*.88)]
        gray = cv2.cvtColor(cv2.resize(crop, (320, 180)), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        if previous is None:
            score = 0.0
        else:
            diff = cv2.absdiff(gray, previous)
            score = float(np.mean(diff > 20))
        samples.append({'time': round(idx / fps, 3), 'motion': round(score, 5)})
        previous = gray
    cap.release()
    if len(samples) < 8:
        raise ValueError('Video is too short for analysis.')
    values = np.array([s['motion'] for s in samples])
    # Video-dependent threshold, bounded to avoid treating camera noise as activity.
    threshold = float(np.clip(np.percentile(values, 40) + .35 * (np.percentile(values, 85)-np.percentile(values, 40)), .018, .20))
    active = values >= threshold
    # Smooth over brief inactive gaps, but do not bridge long breaks.
    step = stride / fps
    gap_samples = max(1, round(1.25 / step))
    for i in range(1, len(active)-1):
        if not active[i] and any(active[max(0, i-gap_samples):i]) and any(active[i+1:min(len(active), i+gap_samples+1)]):
            active[i] = True
    intervals = []
    start = None
    for i, flag in enumerate(list(active) + [False]):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            a = max(0.0, samples[start]['time'] - .5)
            b = min(duration, samples[i-1]['time'] + step + .5)
            if b-a >= min_rally_seconds:
                intervals.append([a,b])
            start = None
    # Merge near-adjacent candidate segments.
    merged = []
    for a,b in intervals:
        if merged and a-merged[-1][1] < 1.5:
            merged[-1][1] = b
        else:
            merged.append([a,b])
    rallies = [{'id':i+1, 'start':round(a,2), 'end':round(b,2), 'duration':round(b-a,2), 'status':'unreviewed'} for i,(a,b) in enumerate(merged)]
    return {'duration':round(duration,2), 'fps':round(fps,2), 'sample_fps':round(1/step,2), 'threshold':round(threshold,4), 'rallies':rallies, 'motion_series':samples, 'method':'motion_heuristic_v1', 'limitations':'Candidate active-play intervals only. Camera movement, people walking, lighting, and editing can cause errors. No ball tracking, shot identification, point winner, or expected points calculated.'}
