import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class LapSRN(nn.Module):
    def __init__(self, upscale_factor=4, num_channels=3):
        super(LapSRN, self).__init__()
        self.upscale_factor = upscale_factor
        self.num_channels = num_channels

        # 特征提取模块
        self.init_feature = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # 残差特征组（每层 20 个残差块）
        res_blocks = []
        for _ in range(20):
            res_blocks.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            res_blocks.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.res_blocks = nn.Sequential(*res_blocks)

        # 上采样 ×2 分支（可级联实现 ×4 / ×8）
        self.upscale2x = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.pred_res2x = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)
        self.pred_hr2x = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)

        # ×4 上采样
        if upscale_factor >= 4:
            self.upscale4x = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
            self.pred_res4x = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)
            self.pred_hr4x = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)

        # ×8 上采样
        if upscale_factor >= 8:
            self.upscale8x = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
            self.pred_res8x = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)
            self.pred_hr8x = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 初始特征
        f = self.init_feature(x)
        f = self.res_blocks(f)

        # 第一次上采样 ×2
        up2x = self.upscale2x(f)
        res2x = self.pred_res2x(up2x)
        bicubic2x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        hr2x = bicubic2x + res2x

        if self.upscale_factor == 2:
            return hr2x

        # 第二次上采样 ×4
        up4x = self.upscale4x(up2x)
        res4x = self.pred_res4x(up4x)
        bicubic4x = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)
        hr4x = bicubic4x + res4x

        if self.upscale_factor == 4:
            return hr4x

        # 第三次上采样 ×8
        up8x = self.upscale8x(up4x)
        res8x = self.pred_res8x(up8x)
        bicubic8x = F.interpolate(x, scale_factor=8, mode='bicubic', align_corners=False)
        hr8x = bicubic8x + res8x

        return hr8x

# ========================================
def img2tensor(img):
    """numpy 转 tensor"""
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor

def tensor2img(tensor):
    """tensor 转 numpy"""
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img

# =================import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

# ===================== 1. LapSRN 核心网络结构====================
class LapSRN(nn.Module):
    def __init__(self, upscale_factor=4, num_channels=3):
        super(LapSRN, self).__init__()
        self.upscale_factor = upscale_factor
        self.num_channels = num_channels

        # 特征提取模块
        self.init_feature = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # 残差特征组（每层 20 个残差块）
        res_blocks = []
        for _ in range(20):
            res_blocks.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            res_blocks.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.res_blocks = nn.Sequential(*res_blocks)

        # 上采样 ×2 分支（可级联实现 ×4 / ×8）
        self.upscale2x = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.pred_res2x = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)
        self.pred_hr2x = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)

        # ×4 上采样
        if upscale_factor >= 4:
            self.upscale4x = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
            self.pred_res4x = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)
            self.pred_hr4x = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)

        # ×8 上采样
        if upscale_factor >= 8:
            self.upscale8x = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
            self.pred_res8x = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)
            self.pred_hr8x = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 初始特征
        f = self.init_feature(x)
        f = self.res_blocks(f)

        # 第一次上采样 ×2
        up2x = self.upscale2x(f)
        res2x = self.pred_res2x(up2x)
        bicubic2x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        hr2x = bicubic2x + res2x

        if self.upscale_factor == 2:
            return hr2x

        # 第二次上采样 ×4
        up4x = self.upscale4x(up2x)
        res4x = self.pred_res4x(up4x)
        bicubic4x = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)
        hr4x = bicubic4x + res4x

        if self.upscale_factor == 4:
            return hr4x

        # 第三次上采样 ×8
        up8x = self.upscale8x(up4x)
        res8x = self.pred_res8x(up8x)
        bicubic8x = F.interpolate(x, scale_factor=8, mode='bicubic', align_corners=False)
        hr8x = bicubic8x + res8x

        return hr8x

# ===================== 2. 图像预处理 / 后处理工具函数 =====================
def img2tensor(img):
    """numpy 转 tensor"""
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor

def tensor2img(tensor):
    """tensor 转 numpy"""
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img

# ===================== 3. 测试代码=====================
if __name__ == "__main__":
    # 1. 配置
    upscale = 4  # 放大倍数：2 / 4 / 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 加载模型
    model = LapSRN(upscale_factor=upscale).to(device)
    model.eval()  # 测试模式

    # 3. 读取图像
    lr_img = cv2.imread("test.jpg")  # 低分辨率输入图
    lr_tensor = img2tensor(lr_img).to(device)

    # 4. 超分辨率推理
    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    # 5. 保存结果
    sr_img = tensor2img(sr_tensor)
    cv2.imwrite(f"LapSRN_x{upscale}.png", sr_img)

    print(f"LapSRN 超分辨率完成！放大倍数 ×{upscale}")=============
if __name__ == "__main__":
    # 1. 配置
    upscale = 4  # 放大倍数：2 / 4 / 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 加载模型
    model = LapSRN(upscale_factor=upscale).to(device)
    model.eval()  # 测试模式

    # 3. 读取图像
    lr_img = cv2.imread("test.jpg")  # 低分辨率输入图
    lr_tensor = img2tensor(lr_img).to(device)

    # 4. 超分辨率推理
    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    # 5. 保存结果
    sr_img = tensor2img(sr_tensor)
    cv2.imwrite(f"LapSRN_x{upscale}.png", sr_img)

    print(f"LapSRN 超分辨率完成！放大倍数 ×{upscale}")