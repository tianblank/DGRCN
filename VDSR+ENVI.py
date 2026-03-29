import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from spectral import envi
import matplotlib.pyplot as plt
import pandas as pd

# ===================== VDSR 模型（20层卷积 + 残差连接） =====================
class VDSR(nn.Module):
    def __init__(self, num_channels=3):
        super(VDSR, self).__init__()
        self.num_channels = num_channels

        # 首层
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        # 中间18层卷积
        body = []
        for _ in range(18):
            body.append(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True))
            body.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*body)

        # 末层
        self.conv_last = nn.Conv2d(64, num_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.body(out)
        out = self.conv_last(out)
        out = torch.add(out, residual)  # 残差
        return out

# ===================== ENVI 数据工具 =====================
def read_envi(hdr_path):
    img = envi.open(hdr_path)
    data = np.array(img.load()).astype(np.float32)
    return data, img.header

def normalize(data):
    return (data - data.min()) / (data.max() - data.min() + 1e-8)

def to_tensor(data):
    return torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).float()

def to_numpy(tensor):
    return tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

def save_envi(hdr, data, save_path):
    meta = hdr.copy()
    meta["lines"], meta["samples"] = data.shape[:2]
    envi.save_image(save_path, data, metadata=meta, force=True)

# ===================== Bicubic 上采样（VDSR前置步骤） =====================
def bicubic_upsample(lr_data, scale):
    h, w, c = lr_data.shape
    lr = to_tensor(lr_data)
    sr = F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)
    return to_numpy(sr)

# ===================== PSNR / SSIM 评价 =====================
def calculate_metrics(gt, sr):
    gt = normalize(gt)
    sr = normalize(sr)
    psnr_list, ssim_list = [], []
    for b in range(gt.shape[-1]):
        g = gt[..., b]
        s = sr[..., b]
        psnr_val = psnr(g, s, data_range=1.0)
        ssim_val = ssim(g, s, data_range=1.0, win_size=min(7, g.shape[0]))
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
    return np.mean(psnr_list), np.mean(ssim_list)

# ===================== 图表输出 =====================
def plot_results(methods, psnrs, ssims):
    plt.figure(figsize=(10,4))
    plt.subplot(121); plt.bar(methods, psnrs); plt.title("PSNR (dB)")
    plt.subplot(122); plt.bar(methods, ssims); plt.title("SSIM")
    plt.tight_layout()
    plt.savefig("vdsr_metrics.png", dpi=300)

def print_table(res):
    df = pd.DataFrame([[k, round(v["PSNR"],4), round(v["SSIM"],4)] for k,v in res.items()],
                      columns=["Method", "PSNR", "SSIM"])
    print("\n" + "="*55)
    print("定量评价表")
    print("="*55)
    print(df.to_string(index=False))
    df.to_csv("vdsr_results.csv", index=False)

# ===================== 主程序 =====================
if __name__ == "__main__":

    LR_PATH = "low_res.hdr"
    GT_PATH = "high_res_gt.hdr"
    UPSCALE = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取数据
    lr_data, hdr = read_envi(LR_PATH)
    gt_data, _ = read_envi(GT_PATH)
    C = lr_data.shape[-1]

    # ---------------- Bicubic ----------------
    bic_data = bicubic_upsample(lr_data, UPSCALE)
    bic_psnr, bic_ssim = calculate_metrics(gt_data, bic_data)

    # ---------------- VDSR ----------------
    model = VDSR(num_channels=C).to(DEVICE).eval()
    with torch.no_grad():
        bic_tensor = to_tensor(normalize(bic_data)).to(DEVICE)
        sr_tensor = model(bic_tensor)
    vdsr_data = to_numpy(sr_tensor)
    vdsr_psnr, vdsr_ssim = calculate_metrics(gt_data, vdsr_data)

    # 保存结果
    save_envi(hdr, bic_data, f"Bicubic_x{UPSCALE}.hdr")
    save_envi(hdr, vdsr_data, f"VDSR_x{UPSCALE}.hdr")

    # 输出结果
    results = {
        f"Bicubic_x{UPSCALE}": {"PSNR": bic_psnr, "SSIM": bic_ssim},
        f"VDSR_x{UPSCALE}": {"PSNR": vdsr_psnr, "SSIM": vdsr_ssim}
    }
    print_table(results)
    plot_results(list(results.keys()), [results[m]["PSNR"] for m in results], [results[m]["SSIM"] for m in results])

    print(f"\n VDSR 超分完成！")
