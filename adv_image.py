import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os

from art.attacks.evasion import DeepFool
from art.estimators.classification import PyTorchClassifier

# ✅ 디바이스 설정 (MPS 또는 CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"✅ Using device: {device}")

# ✅ CNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ✅ 모델 로드 및 설정
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# ✅ ART classifier 래핑
classifier = PyTorchClassifier(
    model=model,
    loss=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=0.001),
    input_shape=(1, 28, 28),
    nb_classes=10,
    device_type="mps" if device.type == "mps" else "cpu"
)

# ✅ MNIST 테스트 데이터 로딩
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

# ✅ DeepFool 인스턴스 생성
deepfool = DeepFool(classifier=classifier)

# ✅ 적대적 샘플 생성
x_all = []
y_all = []

for images, labels in test_loader:
    x_np = images.numpy()
    y_np = labels.numpy()

    x_adv = deepfool.generate(x=x_np)

    x_all.append(torch.tensor(x_adv))
    y_all.append(torch.tensor(y_np))

# ✅ 결과 병합 및 저장
adv_images = torch.cat(x_all, dim=0)
adv_labels = torch.cat(y_all, dim=0)

os.makedirs("adv_samples", exist_ok=True)
torch.save(adv_images, "adv_samples/deepfool_mnist_images.pt")
torch.save(adv_labels, "adv_samples/deepfool_mnist_labels.pt")

print("✅ DeepFool 적대적 샘플 생성 및 저장 완료.")