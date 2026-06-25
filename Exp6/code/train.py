"""
Experiment 6: SegNet Semantic Segmentation on CamVid
"""
import os
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import urllib.request
import zipfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data')
FIGS = os.path.join(BASE, 'figs')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

NUM_CLASSES = 32
IMG_SIZE = (360, 480)


# ── Data ───────────────────────────────────────────────────────

def download_camvid():
    """Download CamVid dataset from a mirror."""
    camvid_dir = os.path.join(DATA_DIR, 'CamVid')
    if os.path.exists(camvid_dir) and len(os.listdir(camvid_dir)) > 0:
        return camvid_dir

    print('Downloading CamVid dataset...')
    url = 'https://github.com/alexgkendall/SegNet-Tutorial/archive/refs/heads/master.zip'
    zip_path = os.path.join(DATA_DIR, 'segnet_tutorial.zip')
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(DATA_DIR)

    extracted = os.path.join(DATA_DIR, 'SegNet-Tutorial-master', 'CamVid')
    if os.path.exists(extracted):
        os.rename(extracted, camvid_dir)

    return camvid_dir


# CamVid color map (32 classes)
CAMVID_COLORS = np.array([
    [128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128],
    [0, 0, 192], [128, 128, 0], [192, 128, 128], [64, 64, 128],
    [64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 0],
    [192, 128, 0], [0, 0, 128], [128, 128, 64], [0, 64, 64],
    [128, 0, 192], [192, 0, 64], [128, 128, 192], [0, 0, 64],
    [192, 0, 128], [128, 64, 64], [64, 192, 0], [0, 128, 64],
    [128, 192, 0], [64, 128, 64], [192, 0, 0], [64, 128, 192],
    [192, 64, 0], [64, 0, 64], [192, 64, 128], [0, 0, 0],
], dtype=np.uint8)

CLASS_NAMES = [
    'Sky', 'Building', 'Pole', 'Road', 'Pavement', 'Tree', 'SignSymbol',
    'Fence', 'Car', 'Pedestrian', 'Bicyclist', 'Void',
    'Column_Pole', 'Road_marking', 'Road_shoulder', 'Sidewalk',
    'Misc_Text', 'Signal', 'Bridge', 'Tunnel', 'Archway',
    'OtherMoving', 'LaneMkgsDriv', 'LaneMkgsNonDriv', 'Animal',
    'SUVPickupTruck', 'TrafficLight', 'Truck_Bus', 'Train',
    'Wall', 'CartLuggPram', 'Child'
]


def color_to_label(mask_img):
    """Convert RGB mask to class index map."""
    mask = np.array(mask_img)
    h, w = mask.shape[:2]
    label = np.zeros((h, w), dtype=np.int64)
    for i, color in enumerate(CAMVID_COLORS):
        match = np.all(mask == color, axis=-1)
        label[match] = i
    return label


class CamVidDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(360, 480)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, img_name)).convert('RGB')

        img = self.transform(img)
        mask = mask.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        label = torch.from_numpy(color_to_label(mask)).long()
        return img, label


# ── SegNet Model ───────────────────────────────────────────────

class SegNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=32):
        super().__init__()
        # Encoder (VGG16-like)
        self.enc1 = self._make_encoder_block(in_channels, 64, 2)
        self.enc2 = self._make_encoder_block(64, 128, 2)
        self.enc3 = self._make_encoder_block(128, 256, 3)
        self.enc4 = self._make_encoder_block(256, 512, 3)
        self.enc5 = self._make_encoder_block(512, 512, 3)

        # Decoder
        self.dec5 = self._make_decoder_block(512, 512, 3)
        self.dec4 = self._make_decoder_block(512, 256, 3)
        self.dec3 = self._make_decoder_block(256, 128, 3)
        self.dec2 = self._make_decoder_block(128, 64, 2)
        self.dec1 = self._make_decoder_block(64, 64, 2)

        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def _make_encoder_block(self, in_ch, out_ch, n_conv):
        layers = []
        for i in range(n_conv):
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _make_decoder_block(self, in_ch, out_ch, n_conv):
        layers = []
        for i in range(n_conv):
            if i == n_conv - 1:
                layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
            else:
                layers.append(nn.Conv2d(in_ch, in_ch, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch if i == n_conv - 1 else in_ch))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x = self.enc1(x)
        x, idx1 = nn.functional.max_pool2d(x, 2, 2, return_indices=True)
        x = self.enc2(x)
        x, idx2 = nn.functional.max_pool2d(x, 2, 2, return_indices=True)
        x = self.enc3(x)
        x, idx3 = nn.functional.max_pool2d(x, 2, 2, return_indices=True)
        x = self.enc4(x)
        x, idx4 = nn.functional.max_pool2d(x, 2, 2, return_indices=True)
        x = self.enc5(x)
        x, idx5 = nn.functional.max_pool2d(x, 2, 2, return_indices=True)

        # Decoder
        x = nn.functional.max_unpool2d(x, idx5, 2, 2, output_size=idx4.shape[2:])
        x = self.dec5(x)
        x = nn.functional.max_unpool2d(x, idx4, 2, 2, output_size=idx3.shape[2:])
        x = self.dec4(x)
        x = nn.functional.max_unpool2d(x, idx3, 2, 2, output_size=idx2.shape[2:])
        x = self.dec3(x)
        x = nn.functional.max_unpool2d(x, idx2, 2, 2, output_size=idx1.shape[2:])
        x = self.dec2(x)
        x = nn.functional.max_unpool2d(x, idx1, 2, 2)
        x = self.dec1(x)

        return self.classifier(x)


# ── Metrics ────────────────────────────────────────────────────

def compute_metrics(pred, target, num_classes):
    """Compute PA, mPA, mIoU."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    # Pixel Accuracy
    pa = (pred_flat == target_flat).sum().item() / len(target_flat)

    # Per-class accuracy and IoU
    class_acc = []
    class_iou = []
    for c in range(num_classes):
        tp = ((pred_flat == c) & (target_flat == c)).sum().item()
        fp = ((pred_flat == c) & (target_flat != c)).sum().item()
        fn = ((pred_flat != c) & (target_flat == c)).sum().item()
        total_c = (target_flat == c).sum().item()
        if total_c > 0:
            class_acc.append(tp / total_c)
        if tp + fp + fn > 0:
            class_iou.append(tp / (tp + fp + fn))

    mpa = np.mean(class_acc) if class_acc else 0.0
    miou = np.mean(class_iou) if class_iou else 0.0
    return pa, mpa, miou


# ── Training ───────────────────────────────────────────────────

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    camvid_dir = download_camvid()

    # Setup datasets
    train_img_dir = os.path.join(camvid_dir, 'train')
    train_mask_dir = os.path.join(camvid_dir, 'trainannot')
    val_img_dir = os.path.join(camvid_dir, 'val')
    val_mask_dir = os.path.join(camvid_dir, 'valannot')
    test_img_dir = os.path.join(camvid_dir, 'test')
    test_mask_dir = os.path.join(camvid_dir, 'testannot')

    train_dataset = CamVidDataset(train_img_dir, train_mask_dir, img_size=IMG_SIZE)
    val_dataset = CamVidDataset(val_img_dir, val_mask_dir, img_size=IMG_SIZE)
    test_dataset = CamVidDataset(test_img_dir, test_mask_dir, img_size=IMG_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')

    # Model
    model = SegNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {param_count:,}')

    # Class weights (inverse frequency)
    criterion = nn.CrossEntropyLoss(ignore_index=11)  # ignore Void class
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    history = {'train_loss': [], 'val_loss': [], 'val_pa': [], 'val_mpa': [],
               'val_miou': [], 'lr': [], 'epoch_time': []}
    best_miou = 0.0
    epochs = 50

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches, t0 = 0.0, 0, time.time()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches
        elapsed = time.time() - t0

        # Validation
        model.eval()
        val_loss_total, val_n = 0.0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                output = model(imgs)
                loss = criterion(output, labels)
                val_loss_total += loss.item()
                val_n += 1
                preds = output.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(labels.cpu().numpy())

        val_loss = val_loss_total / val_n
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        pa, mpa, miou = compute_metrics(all_preds, all_targets, NUM_CLASSES)

        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_loss)
        history['val_pa'].append(pa)
        history['val_mpa'].append(mpa)
        history['val_miou'].append(miou)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['epoch_time'].append(elapsed)

        print(f'Epoch {epoch+1}/{epochs}  train_loss={avg_loss:.4f}  val_loss={val_loss:.4f}  '
              f'PA={pa:.4f}  mPA={mpa:.4f}  mIoU={miou:.4f}  time={elapsed:.1f}s')

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), os.path.join(BASE, 'segnet_best.pth'))

    # Test evaluation
    model.load_state_dict(torch.load(os.path.join(BASE, 'segnet_best.pth'), weights_only=True))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    test_pa, test_mpa, test_miou = compute_metrics(all_preds, all_targets, NUM_CLASSES)
    history['test_pa'] = test_pa
    history['test_mpa'] = test_mpa
    history['test_miou'] = test_miou
    print(f'\nTest Results: PA={test_pa:.4f}  mPA={test_mpa:.4f}  mIoU={test_miou:.4f}')

    with open(os.path.join(BASE, 'all_history.json'), 'w') as f:
        json.dump(history, f)

    print('Done!')


if __name__ == '__main__':
    train()
