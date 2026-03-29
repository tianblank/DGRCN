import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np

# ===================== 数据集定义 =====================
class SRDataset(Dataset):
    def __init__(self, hr_img_paths, upscale_factor=4):
        self.hr_paths = hr_img_paths
        self.upscale = upscale_factor
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        # 读取高分辨率图(HR)
        hr = cv2.imread(self.hr_paths[idx])
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        h, w = hr.shape[:2]
        # 下采样 → 低分辨率(LR)
        lr = cv2.resize(hr, (w//self.upscale, h//self.upscale), interpolation=cv2.INTER_CUBIC)
        # 双三次上采样 → 与HR同尺寸（模型输入）
        lr_up = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)

        # 转Tensor
        hr_tensor = self.transform(hr)
        lr_up_tensor = self.transform(lr_up)
        return lr_up_tensor, hr_tensor

# ===================== 训练配置 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DGRCN(num_layers=20).to(device)
criterion = nn.MSELoss()  # 超分标准损失
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 100

# 替换为高清图像路径
train_paths = ["your_hr_image1.png", "your_hr_image2.png"]
dataset = SRDataset(train_paths)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# ===================== 训练循环 =====================
print("开始训练...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for lr_up, hr in dataloader:
        lr_up, hr = lr_up.to(device), hr.to(device)

        # 前向传播
        sr = model(lr_up)  # 超分输出
        loss = criterion(sr, hr)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

# 保存模型
torch.save(model.state_dict(), "DGRCN_SR.pth")
print("模型训练完成，已保存！")
