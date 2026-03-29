import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== 1. build DGRCN =====================
class DGRCN(nn.Module):
    def __init__(self, num_layers=20, num_channels=64):
        super(DGRCN, self).__init__()
        self.num_layers = num_layers

        # 【浅层特征提取】→ 继承DnCNN首层结构
        self.shallow_conv = nn.Conv2d(
            in_channels=3,    # RGB图像
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
            bias=True
        )
        self.relu = nn.ReLU(inplace=True)

        # 【深度残差层】→ 融合DnCNN (Conv+BN) + VDSR (深度堆叠)
        self.deep_layers = nn.ModuleList()
        for i in range(num_layers):
            self.deep_layers.append(
                nn.Conv2d(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False    # BN层会包含偏置，故关闭
                )
            )
            self.deep_layers.append(nn.BatchNorm2d(num_channels))  # DnCNN核心：BN归一化
            self.deep_layers.append(nn.ReLU(inplace=True))

        # 【重建层】输出高分辨率图像
        self.reconstruction = nn.Conv2d(
            in_channels=num_channels,
            out_channels=3,
            kernel_size=3,
            padding=1,
            bias=True
        )

        # 权重初始化 (适合ReLU)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入：x = 双三次上采样后的低分辨率图像 (B, 3, H, W)
        residual = x  # 保存原始输入（用于全局残差学习 → VDSR核心）

        # 1. 浅层特征提取
        out = self.relu(self.shallow_conv(x))
        shallow_feat = out  # 保存浅层特征（全局残差跳连）

        # 2. 深度残差特征学习
        for layer in self.deep_layers:
            out = layer(out)

        # 3. 【全局残差连接】→ VDSR核心：浅层特征 + 深层特征
        out = torch.add(out, shallow_feat)

        # 4. 图像重建
        out = self.reconstruction(out)

        # 5. 最终全局残差 (输出 = 预测残差 + 输入上采样图)
        out = torch.add(out, residual)

        return out

# ===================== 2. 测试模型 =====================
if __name__ == '__main__':
    # 初始化模型（20层，适配超分任务）
    model = DGRCN(num_layers=20)
    print(model)

    # 测试前向传播
    test_input = torch.randn(1, 3, 256, 256)  # (batch, channel, H, W)
    with torch.no_grad():
        output = model(test_input)
    print(f"输入尺寸: {test_input.shape}")
    print(f"输出尺寸: {output.shape}")  # 输出与输入同尺寸 → 超分完成
