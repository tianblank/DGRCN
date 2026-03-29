def eval_sr(model_path, lr_img_path, save_path, upscale_factor=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DGRCN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 读取低分辨率图
    lr = cv2.imread(lr_img_path)
    h, w = lr.shape[:2]
    # 双三次上采样
    lr_up = cv2.resize(lr, (w*upscale_factor, h*upscale_factor), interpolation=cv2.INTER_CUBIC)
    lr_rgb = cv2.cvtColor(lr_up, cv2.COLOR_BGR2RGB)

    # 预处理
    transform = transforms.ToTensor()
    lr_tensor = transform(lr_rgb).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    
    # 后处理
    sr_img = sr_tensor.squeeze().cpu().permute(1,2,0).numpy()
    sr_img = (sr_img * 255.0).clip(0, 255).astype(np.uint8)
    sr_img = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, sr_img)
    print(f"超分完成，保存至：{save_path}")

# 调用示例
eval_sr("DGRCN_SR.pth", "lr_test.png", "sr_result.png")
