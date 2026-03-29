import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from spectral import envi

# ===================== 1. LapSRN 模型 =====================
class LapSRN(nn.Module):
    def __init__(self, upscale_factor=4, num_channels=3):
        super(LapSRN, self).__init__()
        self.upscale_factor = upscale_factor
        self.num_channels = num_channels

        # 特征提取
        self.init_feature = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # 20 层残差特征
        res_blocks = []
        for _ in range(20):
            res_blocks.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            res_blocks.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.res_blocks = nn.Sequential(*res_blocks)

        # ×2 上采样
        self.upscale2x = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.pred_res2x = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)

        # ×4 上采样
        if upscale_factor >= 4:
            self.upscale4x = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
            self.pred_res4x = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)

        # ×8 上采样
        if upscale_factor >= 8:
            self.upscale8x = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
            self.pred_res8x = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        f = self.init_feature(x)
        f = self.res_blocks(f)

        # ×2
        up2 = self.upscale2x(f)
        res2 = self.pred_res2x(up2)
        bicubic2 = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        hr2 = bicubic2 + res2

        if self.upscale_factor == 2:
            return hr2

        # ×4
        up4 = self.upscale4x(up2)
        res4 = self.pred_res4x(up4)
        bicubic4 = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)
        hr4 = bicubic4 + res4

        if self.upscale_factor == 4:
            return hr4

        # ×8
        up8 = self.upscale8x(up4)
        res8 = self.pred_res8x(up8)
        bicubic8 = F.interpolate(x, scale_factor=8, mode='bicubic', align_corners=False)
        hr8 = bicubic8 + res8

        return hr8

# ===================== 2. ENVI函数 =====================
def read_envi_data(hdr_path):
    """读取 ENVI .hdr + .dat 数据 → 返回 numpy 数组 + 头文件信息"""
    img = envi.open(hdr_path)
    data = img.load()  # (行, 列, 波段)
    hdr_info = img.header
    return data, hdr_info

def envi_data_preprocess(data):
    """数据归一化 → 适合神经网络输入"""
    data = data.astype(np.float32)
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    return data

def numpy2tensor(data):
    """(H, W, C) → (1, C, H, W) PyTorch 张量"""
    tensor = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0)
    return tensor

def tensor2numpy(tensor):
    """张量转回 (H, W, C) numpy 数组"""
    data = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    data = np.clip(data, 0, 1)
    return data

def save_envi_superres(hdr_original, sr_data, save_path):
    """
    保存超分辨率后的 ENVI 数据
  
    """
    lines, samples, bands = sr_data.shape
    metadata = hdr_original.copy()
    metadata['lines'] = lines
    metadata['samples'] = samples
    metadata['data type'] = 4  # float32 通用格式
    envi.save_image(save_path, sr_data, metadata=metadata, force=True)
    print(f" 超分后 ENVI 数据已保存：{save_path}")

# ===================== 3. 主函数图像超分辨率 =====================
if __name__ == "__main__":

    HDR_PATH = "your_data.hdr"       # 输入 ENVI 头文件
    SAVE_PATH = "sr_result.hdr"      # 输出超分结果
    UPSCALE = 4                      # 放大倍数：2 / 4 / 8
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 读取原始数据
    lr_data, hdr = read_envi_data(HDR_PATH)
    H, W, C = lr_data.shape
    print(f" 原始数据尺寸：{H} × {W} × {C} 波段")

    # 2. 预处理
    lr_data_norm = envi_data_preprocess(lr_data)
    lr_tensor = numpy2tensor(lr_data_norm).to(DEVICE)

    # 3. 加载 LapSRN
    model = LapSRN(upscale_factor=UPSCALE, num_channels=C).to(DEVICE)
    model.eval()

    # 4. 超分辨率推理
    print(" 正在运行 LapSRN 超分辨率...")
    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    # 5. 转回 numpy
    sr_data = tensor2numpy(sr_tensor)
    print(f" 超分后尺寸：{sr_data.shape[0]} × {sr_data.shape[1]} × {sr_data.shape[2]} 波段")

    # 6. 保存为 ENVI 格式（保留所有元信息）
    save_envi_superres(hdr, sr_data, SAVE_PATH)
