"""
Experiment 5: YOLOv5 Object Detection
Train YOLOv5 on a custom dataset (COCO128 subset) and evaluate detection performance.
"""
import os
import json
import time
import torch
import numpy as np
from pathlib import Path

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(BASE, 'figs')
os.makedirs(FIGS, exist_ok=True)


def train_yolov5():
    from ultralytics import YOLO

    # Use YOLOv5s pretrained model and fine-tune on COCO128
    model = YOLO('yolov5s.pt')

    # Train on COCO128 (auto-downloads)
    results = model.train(
        data='coco128.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        project=os.path.join(BASE, 'runs'),
        name='yolov5s_coco128',
        exist_ok=True,
        plots=True,
        save=True,
        device=0 if torch.cuda.is_available() else 'cpu',
    )

    # Validate
    val_results = model.val(
        data='coco128.yaml',
        project=os.path.join(BASE, 'runs'),
        name='yolov5s_val',
        exist_ok=True,
    )

    # Extract metrics
    history = {
        'map50': float(val_results.box.map50),
        'map50_95': float(val_results.box.map),
        'precision': float(val_results.box.mp),
        'recall': float(val_results.box.mr),
    }

    # Inference on sample images
    predict_results = model.predict(
        source=os.path.join(BASE, 'runs', 'yolov5s_coco128', 'val_batch0_labels.jpg'),
        save=True,
        project=os.path.join(BASE, 'runs'),
        name='predict',
        exist_ok=True,
    )

    # Save history
    # Parse training CSV for loss curves
    csv_path = os.path.join(BASE, 'runs', 'yolov5s_coco128', 'results.csv')
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        history['epochs'] = list(range(1, len(df) + 1))
        for col in ['train/box_loss', 'train/cls_loss', 'train/dfl_loss',
                    'metrics/precision(B)', 'metrics/recall(B)',
                    'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
                    'val/box_loss', 'val/cls_loss', 'val/dfl_loss', 'lr/pg0']:
            if col in df.columns:
                history[col] = df[col].tolist()

    with open(os.path.join(BASE, 'training_history.json'), 'w') as f:
        json.dump(history, f)

    print(f"\nFinal Results:")
    print(f"  mAP@0.5: {history['map50']*100:.1f}%")
    print(f"  mAP@0.5:0.95: {history['map50_95']*100:.1f}%")
    print(f"  Precision: {history['precision']*100:.1f}%")
    print(f"  Recall: {history['recall']*100:.1f}%")

    return history


def detect_samples():
    """Run detection on various sample images."""
    from ultralytics import YOLO
    import urllib.request

    model_path = os.path.join(BASE, 'runs', 'yolov5s_coco128', 'weights', 'best.pt')
    if not os.path.exists(model_path):
        model_path = 'yolov5s.pt'

    model = YOLO(model_path)

    # Download sample images
    sample_dir = os.path.join(BASE, 'samples')
    os.makedirs(sample_dir, exist_ok=True)

    sample_urls = {
        'bus': 'https://ultralytics.com/images/bus.jpg',
        'zidane': 'https://ultralytics.com/images/zidane.jpg',
    }

    for name, url in sample_urls.items():
        path = os.path.join(sample_dir, f'{name}.jpg')
        if not os.path.exists(path):
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"Failed to download {name}: {e}")

    # Run inference
    results = model.predict(
        source=sample_dir,
        save=True,
        project=os.path.join(BASE, 'runs'),
        name='samples_detect',
        exist_ok=True,
        conf=0.25,
    )

    # Collect detection stats
    det_stats = []
    for r in results:
        det_stats.append({
            'image': os.path.basename(r.path),
            'num_objects': len(r.boxes),
            'classes': [r.names[int(c)] for c in r.boxes.cls],
            'confidences': r.boxes.conf.tolist(),
        })

    with open(os.path.join(BASE, 'detection_results.json'), 'w') as f:
        json.dump(det_stats, f, ensure_ascii=False)

    return det_stats


if __name__ == '__main__':
    print("=" * 60)
    print("YOLOv5 Object Detection Experiment")
    print("=" * 60)

    history = train_yolov5()
    print("\nRunning inference on samples...")
    detect_samples()
    print("\nDone!")
