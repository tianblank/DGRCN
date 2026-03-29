import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from spectral import envi

# ===================== 1. ENVI 数据读取函数 =====================
def read_envi(hdr_path):
    """读取 ENVI 数据"""
    img = envi.open(hdr_path)
    data = img.load()  # shape: [行, 列, 波段]
    data = np.array(data).astype(np.float32)
    return data

# ===================== 2. PSNR / SSIM 函数 =====================
def compute_metrics(img_gt, img_sr):
    """
    计算 高光谱/多光谱 图像的 PSNR 和 SSIM
    :param img_gt: 高分辨率真值图像 (H, W, C)
    :param img_sr: 超分辨率结果图像 (H, W, C)
    :return: 平均PSNR, 平均SSIM, 各波段PSNR, 各波段SSIM
    """
    # 数据归一化到 [0, 1]
    img_gt = (img_gt - img_gt.min()) / (img_gt.max() - img_gt.min() + 1e-8)
    img_sr = (img_sr - img_sr.min()) / (img_sr.max() - img_sr.min() + 1e-8)

    h, w, c = img_gt.shape
    psnr_list = []
    ssim_list = []

    # 逐波段计算
    for i in range(c):
        gt = img_gt[..., i]
        sr = img_sr[..., i]

        # PSNR
        psnr_val = psnr(gt, sr, data_range=1.0)

        # SSIM
        win_size = min(7, h, w)  # 自动窗口大小
        ssim_val = ssim(
            gt, sr,
            data_range=1.0,
            win_size=win_size,
            channel_axis=None  # 单波段
        )

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    # 求平均值
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    return avg_psnr, avg_ssim, psnr_list, ssim_list

# ===================== 3. 主程序定量评价 =====================
if __name__ == "__main__":

    GT_HDR = "high_res_gt.hdr"    # 高分辨率真值图像
    SR_HDR = "sr_result.hdr"      # 超分结果（LapSRN输出）

    # 读取数据
    img_gt = read_envi(GT_HDR)
    img_sr = read_envi(SR_HDR)

    if img_gt.shape != img_sr.shape:
        raise ValueError(f"尺寸不匹配！真值={img_gt.shape}, 超分结果={img_sr.shape}")

    # 计算指标
    avg_psnr, avg_ssim, psnr_list, ssim_list = compute_metrics(img_gt, img_sr)

    # ===================== 格式输出 =====================
    print("="*60)
    print("           超分辨率定量评价结果（PSNR / SSIM）")
    print("="*60)
    print(f"图像尺寸：{img_gt.shape[0]} × {img_gt.shape[1]} | 波段数：{img_gt.shape[2]}")
    print(f"平均 PSNR：{avg_psnr:.4f} dB")
    print(f"平均 SSIM：{avg_ssim:.4f}")
    print("="*60)

    # 可选：输出前5个波段指标
    print("\n前 5 波段详细指标：")
    for i in range(min(5, len(psnr_list))):
        print(f"波段 {i+1:2d} | PSNR: {psnr_list[i]:.4f} | SSIM: {ssim_list[i]:.4f}")
