import cv2
import numpy as np
import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import apply_rf
import numpy as np
import cv2
import glob
from model.team02_dequantization_net import Dequantization_net
from model.team02_hallucination_net import HDRAutoencoder
from model.team02_linearization_net import LinearizationNet
def preprocess(img_path, padding=32):
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    print(H,W)
    new_H, new_W=H, W
    ldr_val = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Resize to multiple of 64
    if H % 64 != 0 or W % 64 != 0:
        new_H = ((H + 63) // 64) * 64
        new_W = ((W + 63) // 64) * 64
        ldr_val = cv2.resize(ldr_val, (new_W, new_H), interpolation=cv2.INTER_CUBIC)

    # Add symmetric padding
    ldr_val = np.pad(ldr_val, ((padding, padding), (padding, padding), (0, 0)), mode='symmetric')

    # To tensor
    ldr_tensor = torch.from_numpy(ldr_val).float().permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return ldr_tensor, (H, W), (new_H, new_W), img


def postprocess(hdr_tensor, original_shape, padded_shape, padding=32):
    H, W = original_shape
    new_H, new_W = padded_shape

    hdr_np = hdr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    hdr_np = hdr_np[padding:-padding, padding:-padding]

    if new_H != H or new_W != W:
        hdr_np = cv2.resize(hdr_np, (W, H), interpolation=cv2.INTER_CUBIC)

    return (hdr_np * 255).astype(np.float32)  # 或者保留 float32 输出 .hdr 文件

def build_graph(ldr, is_training, models):
    """
    ldr: [b, h, w, c]
    is_training: bool
    models: {'deq': ..., 'lin': ..., 'hal': ...}
    """
    B, C,H,W = ldr.shape

    # --- Dequantization Net ---
    dequant_model = models['deq']
    C_pred = dequant_model(ldr)
    C_pred = torch.clamp(C_pred, 0.0, 1.0)  # _clip

    # --- Linearization Net ---
    lin_model = models['lin']
    pred_invcrf = lin_model(C_pred)
    B_pred = apply_rf(C_pred.cpu(), pred_invcrf.cpu()).to('cuda')

    # --- Alpha mask ---
    alpha = B_pred.max(dim=1).values # [B, H, W]
    thr = 0.12
    alpha = torch.clamp(alpha - 1.0 + thr, 0.0, 1.0) / (thr + 1e-6)
    alpha = alpha.unsqueeze(1).expand(B,3,H, W)

    # --- Hallucination Net ---
    hal_model = models['hal']
    y_predict_test,_ = hal_model(B_pred)
    y_predict_test = F.relu(y_predict_test)

    A_pred = B_pred + alpha * y_predict_test

    return A_pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--ckpt_path_deq', type=str, default="model_zoo/team02_dequantization_net.ckpt")
    parser.add_argument('--ckpt_path_lin', type=str, default="model_zoo/team02_linearization_net.ckpt")
    parser.add_argument('--ckpt_path_hal', type=str, default="model_zoo/team02_hallucination_net.ckpt")
    parser.add_argument('--test_imgs', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    deq_model = Dequantization_net().to(device)
    lin_model = LinearizationNet().to(device)
    hal_model = HDRAutoencoder().to(device)

    # 加载权重
    deq_model.load_state_dict(torch.load(args.ckpt_path_deq))
    lin_model.load_state_dict(torch.load(args.ckpt_path_lin))
    hal_model.load_state_dict(torch.load(args.ckpt_path_hal))

    deq_model.eval()
    lin_model.eval()
    hal_model.eval()

    models = {
        'deq': deq_model,
        'lin': lin_model,
        'hal': hal_model
    }

    os.makedirs(args.output_path, exist_ok=True)

    test_paths = glob.glob(os.path.join(args.test_imgs, '*.png')) + \
                 glob.glob(os.path.join(args.test_imgs, '*.jpg'))

    for path in sorted(test_paths):
        print(f'Processing {path}')
        input_tensor, original_shape, padded_shape, raw_img = preprocess(path)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output_tensor = build_graph(input_tensor, is_training=False, models=models)

        output_hdr = postprocess(output_tensor, original_shape, padded_shape)

        filename = os.path.splitext(os.path.basename(path))[0]
        output_path = os.path.join(args.output_path, f"{filename}.hdr")
        cv2.imwrite(output_path, output_hdr[..., ::-1])  # RGB -> BGR
        print(f'Saved to {output_path}')

if __name__ == '__main__':
    main()