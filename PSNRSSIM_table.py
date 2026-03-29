import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from spectral import envi
import cv2
import pandas as pd

# ===================== 1. LapSRN 模型 =====================
class LapSRN(nn.Module):
    def __init__(self, upscale_factor=4, num_channels=3):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.init_feature = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        res_blocks = []
        for _ in range(20):
            res_blocks.append(nn.Conv2d(64, 64, 3, padding=1))
            res_blocks.append(nn.LeakyReLU(0.2, inplace=True))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.upscale2x = nn.ConvTranspose2d(64, 64, 4, 2, padding=1)
        self.pred_res2x = nn.Conv2d(64, num_channels, 3, padding=1)
        if upscale_factor >= 4:
            self.upscale4x = nn.ConvTranspose2d(64, 64, 4, 2, padding=1)
            self.pred_res4x = nn.Conv2d(64, num_channels, 3, padding=1)
        if upscale_factor >= 8:
            self.upscale8x = nn.ConvTranspose2d(64, 64, 4, 2, padding=1)
            self.pred_res8x = nn.Conv2d(64, num_channels, 3, padding=1)

    def forward(self, x):
        f = self.init_feature(x)
        f = self.res_blocks(f)
        up2 = self.upscale2x(f)
        hr2 = F.interpolate(x, scale_factor=2, mode='bicubic') + self.pred_res2x(up2)
        if self.upscale_factor == 2: return hr2
        up4 = self.upscale4x(up2)
        hr4 = F.interpolate(x, scale_factor=4, mode='bicubic') + self.pred_res4x(up4)
        if self.upscale_factor == 4: return hr4
        up8 = self.upscale8x(up4)
        hr8 = F.interpolate(x, scale_factor=8, mode='bicubic') + self.pred_res8x(up8)
        return hr8

# ===================== 2. ENVI =====================
def read_envi(hdr_path):
    img = envi.open(hdr_path)
    data = np.array(img.load()).astype(np.float32)
    return data, img.header

def norm_data(data):
    return (data - data.min()) / (data.max() - data.min() + 1e-8)

def tensorize(data):
    return torch.from_numpy(data).permute(2,0,1).unsqueeze(0)

def detensor(tensor):
    return tensor.squeeze(0).permute(1,2,0).cpu().detach().numpy()

def save_envi(hdr, data, path):
    meta = hdr.copy()
    meta['lines'], meta['samples'] = data.shape[:2]
    envi.save_image(path, data, metadata=meta, force=True)

# ===================== 3. Bicubic 上采样 =====================
def bicubic_upscale(lr_data, scale):
    h, w, c = lr_data.shape
    lr_tensor = tensorize(lr_data)
    sr = F.interpolate(lr_tensor, scale_factor=scale, mode='bicubic').squeeze()
    return sr.permute(1,2,0).numpy()

# ===================== 4. PSNR / SSIM 评价 =====================
def calc_metrics(gt, sr):
    gt, sr = norm_data(gt), norm_data(sr)
    psnr_list, ssim_list = [], []
    for i in range(gt.shape[-1]):
        g, s = gt[...,i], sr[...,i]
        psnr_list.append(psnr(g, s, data_range=1))
        ssim_list.append(ssim(g, s, data_range=1, win_size=min(7, g.shape[0])))
    return np.mean(psnr_list), np.mean(ssim_list), psnr_list, ssim_list

# ===================== 5. 表格 & 绘图 =====================
def plot_metrics(results):
    methods = list(results.keys())
    psnrs = [results[m]['PSNR'] for m in methods]
    ssims = [results[m]['SSIM'] for m in methods]
    plt.figure(figsize=(10,4))
    plt.subplot(121); plt.bar(methods, psnrs); plt.title('PSNR (dB)')
    plt.subplot(122); plt.bar(methods, ssims); plt.title('SSIM')
    plt.tight_layout(); plt.savefig('metrics_plot.png', dpi=300)
    plt.close()
    print("✅ 指标图已保存：metrics_plot.png")

def print_table(results):
    df = pd.DataFrame([
        [m, round(results[m]['PSNR'],4), round(results[m]['SSIM'],4)]
        for m in results
    ], columns=['方法', 'PSNR', 'SSIM'])
    print("\n" + "="*50)
    print("📊 定量评价表格")
    print("="*50)
    print(df.to_string(index=False))
    df.to_csv('metrics_table.csv', index=False)
    print("✅ 表格已保存：metrics_table.csv")

# ===================== 6. 主实验 =====================
if __name__ == "__main__":

    LR_HDR = "low_res.hdr"      # 低分辨率输入
    GT_HDR = "high_res_gt.hdr"  # 高分辨率真值
    SCALE = 4                   # 放大倍数
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取数据
    lr_data, hdr = read_envi(LR_HDR)
    gt_data, _ = read_envi(GT_HDR)
    C = lr_data.shape[-1]

    # 1. Bicubic 结果
    print(" 运行 Bicubic 上采样...")
    bic_data = bicubic_upscale(lr_data, SCALE)
    bic_psnr, bic_ssim, _, _ = calc_metrics(gt_data, bic_data)

    # 2. LapSRN 结果
    print(" 运行 LapSRN 超分...")
    model = LapSRN(SCALE, C).to(DEVICE).eval()
    with torch.no_grad():
        sr_tensor = model(tensorize(norm_data(lr_data)).to(DEVICE))
    sr_data = detensor(sr_tensor)
    sr_psnr, sr_ssim, _, _ = calc_metrics(gt_data, sr_data)

    # 保存结果
    save_envi(hdr, sr_data, f"LapSRN_x{SCALE}.hdr")
    save_envi(hdr, bic_data, f"Bicubic_x{SCALE}.hdr")

    # 输出论文结果
    results = {
        f'Bicubic_x{SCALE}': {'PSNR': bic_psnr, 'SSIM': bic_ssim},
        f'LapSRN_x{SCALE}': {'PSNR': sr_psnr, 'SSIM': sr_ssim}
    }
    print_table(results)
    plot_metrics(results)
    print("\n 完成！所有文件已保存！")
