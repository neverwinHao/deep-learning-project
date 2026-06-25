import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

batch_size = 64
lr = 0.001
epochs = 15

print('Loading MNIST...')
transform = transforms.Compose([transforms.ToTensor()])

train_data = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='./data/', train=False, transform=transform, download=True)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

print(f'Train: {len(train_data)}, Test: {len(test_data)}')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc = nn.Linear(32 * 7 * 7, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


model = CNN().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

train_losses, test_losses = [], []
train_accs, test_accs = [], []
best_acc = 0

for epoch in range(epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for x, y in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
        x, y = x.to(device), y.to(device)
        
        out = model(x)
        loss = loss_fn(out, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = torch.max(out, 1)[1]
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    train_loss = total_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            val_loss += loss.item()
            pred = torch.max(out, 1)[1]
            val_correct += (pred == y).sum().item()
            val_total += y.size(0)
    
    test_loss = val_loss / len(test_loader)
    test_acc = val_correct / val_total
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
    print(f'Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}')
    
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'Save model, Acc: {test_acc:.4f}')

model.load_state_dict(torch.load('best_model.pth', weights_only=False))

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
epochs_range = range(1, len(train_losses) + 1)
colors = {'train': '#3498db', 'test': '#e74c3c', 'target': '#f39c12'}

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs_range, train_losses, 'o-', lw=2.5, ms=6, label='Train', color=colors['train'])
ax.plot(epochs_range, test_losses, 's-', lw=2.5, ms=6, label='Test', color=colors['test'])
ax.fill_between(epochs_range, train_losses, alpha=0.1, color=colors['train'])
ax.fill_between(epochs_range, test_losses, alpha=0.1, color=colors['test'])
ax.set(xlabel='Epoch', ylabel='Loss', title='Training & Testing Loss')
ax.legend(frameon=True, fancybox=True, shadow=True)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=200, bbox_inches='tight')
print('Saved loss_curve.png')
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs_range, train_accs, 'o-', lw=2.5, ms=6, label='Train', color=colors['train'])
ax.plot(epochs_range, test_accs, 's-', lw=2.5, ms=6, label='Test', color=colors['test'])
ax.axhline(y=0.98, color=colors['target'], ls='--', lw=2, label='Target (98%)')
ax.fill_between(epochs_range, train_accs, alpha=0.1, color=colors['train'])
ax.fill_between(epochs_range, test_accs, alpha=0.1, color=colors['test'])
ax.set(xlabel='Epoch', ylabel='Accuracy', title='Training & Testing Accuracy')
ax.set_ylim([0.9, 1.005])
ax.legend(frameon=True, fancybox=True, shadow=True, loc='lower right')
plt.tight_layout()
plt.savefig('accuracy_curve.png', dpi=200, bbox_inches='tight')
print('Saved accuracy_curve.png')
plt.show()

test_images, test_labels = next(iter(test_loader))
test_images = test_images[:16].to(device)
test_labels = test_labels[:16]
with torch.no_grad():
    pred_labels = torch.max(model(test_images), 1)[1].cpu()

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle('Test Predictions', fontsize=16, fontweight='bold')
for i, ax in enumerate(axes.flat):
    ax.imshow(test_images[i].cpu().squeeze(), cmap='gray')
    correct = test_labels[i] == pred_labels[i]
    color = '#2ecc71' if correct else '#e74c3c'
    ax.set_title(f'True:{test_labels[i]} Pred:{pred_labels[i]}', fontsize=10, color=color, fontweight='bold')
    ax.axis('off')
plt.tight_layout()
plt.savefig('predictions.png', dpi=200, bbox_inches='tight')
print('Saved predictions.png')
plt.show()

model.eval()
all_pred, all_label = [], []
with torch.no_grad():
    for x, y in test_loader:
        pred = torch.max(model(x.to(device)), 1)[1]
        all_pred.extend(pred.cpu().numpy())
        all_label.extend(y.numpy())

cm = confusion_matrix(all_label, all_pred)
fig, ax = plt.subplots(figsize=(9, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=range(10), yticklabels=range(10),
            linewidths=0.5, linecolor='white', square=True,
            cbar_kws={'label': 'Count', 'shrink': 0.8})
ax.set(xlabel='Predicted', ylabel='True', title='Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=200, bbox_inches='tight')
print('Saved confusion_matrix.png')
plt.show()

print('\n' + '='*50)
print('Classification Report')
print('='*50)
print(classification_report(all_label, all_pred, target_names=[str(i) for i in range(10)]))
print(f'Best Test Accuracy: {best_acc*100:.2f}%')