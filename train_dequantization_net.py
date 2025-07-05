import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from io import BytesIO
import numpy as np
import logging
import argparse
import os
from datetime import datetime
from tqdm import tqdm
# 假设这些模块已适配为 PyTorch 兼容
from util import get_tensor_shape, apply_rf, get_l2_loss_with_mask
from dataset import get_train_dataset,hdr_collate_fn
from model.team02_dequantization_net import Dequantization_net
from model.team02_hallucination_net import HDRAutoencoder
from model.team02_linearization_net import LinearizationNet
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--it_num', type=int, default=100000)  # 
parser.add_argument('--logdir_path', type=str, required=True)
parser.add_argument('--hdr_prefix', type=str, required=True)
parser.add_argument('--resume', type=str, default=None, help="Path to model checkpoint")
ARGS = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
epsilon = 0.001

# Clip 函数
def _clip(x):
    return torch.clamp(x, 0, 1)

# JPEG 压缩模拟
def apply_jpeg_compression(tensor, quality_list):
    transform = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    result = []
    for i in range(tensor.shape[0]):
        img = tensor[i].cpu().detach()
        img = img.permute(1, 2, 0).numpy()
        pil_img = Image.fromarray(img.astype(np.uint8), mode='RGB')
        img_data = BytesIO()
        pil_img.save(img_data, format='JPEG', quality=quality_list[i])
        pil_img = Image.open(img_data)
        img_tensor = to_tensor(pil_img).to(device)* 255.0
        result.append(img_tensor)
    return torch.stack(result)

# PSNR & L2 Loss
def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 20.0 * torch.log10(torch.tensor(1.0)) - 10.0 * torch.log10(mse)

# 构建训练步骤
def train_step(hdr, crf, t, model, optimizer, device="cuda"):
    
    

    # HDR * t
    b, c, h, w = hdr.shape
    hdr=hdr.permute(0, 2, 3, 1)
    _hdr_t = hdr * t.view(b, 1, 1, 1)

    # 添加噪声
    sigma_s = 0.08 / 6 * torch.rand((b, 1, 1, 3))
    sigma_c = 0.005 * torch.rand((b, 1, 1, 3))
    noise_s_map = sigma_s * _hdr_t
    noise_s = torch.randn_like(_hdr_t) * noise_s_map
    temp_x = _hdr_t + noise_s
    noise_c = sigma_c * torch.randn_like(temp_x)
    temp_x += noise_c
    _hdr_t = torch.relu(temp_x)

    clipped_hdr_t = _clip(_hdr_t)

    # 应用 CRF
    ldr = apply_rf(clipped_hdr_t, crf)
    ldr=ldr.permute(0, 3, 1, 2)

    # 量化并压缩
    quantized_hdr = torch.round(ldr * 255.0).byte()
    qualities = [(i % ARGS.batch_size) // (ARGS.batch_size - 1) * 10 + 90 for i in range(ARGS.batch_size)]
    jpeg_img = apply_jpeg_compression(quantized_hdr, qualities)
    jpeg_img_float = jpeg_img.to(dtype=torch.float32)/255
    
    # Loss mask
    # gray = torch.mean(jpeg_img_float, dim=1, keepdim=True)
    rgb_to_gray_transform = transforms.Grayscale(num_output_channels=1)
    gray = rgb_to_gray_transform(jpeg_img)
    
    over_exposed = torch.ge(gray, 249).float()
    under_exposed = torch.le(gray, 6).float()
    over_exposed = torch.sum(over_exposed, dim=[2, 3], keepdim=True)
    under_exposed = torch.sum(under_exposed, dim=[2, 3], keepdim=True)
    extreme_cases = torch.logical_or(over_exposed > 256*256*0.5, under_exposed > 256*256*0.5)
    loss_mask = (~extreme_cases).float().to(device)
    
    ldr=ldr.to(device)
    pred = _clip(model(jpeg_img_float.to(device)))
    loss = get_l2_loss_with_mask(pred, ldr)
    lossmask = torch.mean(loss * loss_mask)
    optimizer.zero_grad()
    lossmask.backward()
    optimizer.step()
    loss=torch.mean(loss)
    psnr = compute_psnr(pred, ldr)
    psnr_no_q = compute_psnr(jpeg_img_float, ldr)

    return loss.item(),lossmask.item(), psnr.item(), psnr_no_q.item()

# 主函数
def main():
    logger.info("Start training...")
    
    # 初始化模型
    model = Dequantization_net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    
    # Resume from checkpoint
    if ARGS.resume:
        logger.info(f"Resuming from {ARGS.resume}")
        model.load_state_dict(torch.load(ARGS.resume))

    # 数据集
    # dataset = RandDatasetReader(get_train_dataset(ARGS.hdr_prefix), ARGS.batch_size)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total trainable parameters: {total_params}")

    # Summary writer
    logdir = os.path.join(ARGS.logdir_path, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logdir, exist_ok=True)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=logdir)
        logger.info("Start writer...")
    except:
        logger.warning("TensorBoard not available.")

    train_dataset = get_train_dataset(ARGS.hdr_prefix)

    # 调试数据集内容
    # for i in range(min(5, len(train_dataset))):
    print(len(train_dataset))  # 看是否能正常输出每个样本

    # 简化 dataloader
    # train_loader = DataLoader(train_dataset, batch_size=ARGS.batch_size, shuffle=True, num_workers=8,collate_fn=hdr_collate_fn)
    # print(len(train_loader))
    # # 开始训练循环
    # for it in range(ARGS.it_num):
    #     logger.info(f"Start train iteration {it}")
    #     for data in train_loader:
    #         logger.info("Batch loaded successfully")
    #         break
    #     break
    train_loader = DataLoader(
        train_dataset,
        batch_size=ARGS.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=hdr_collate_fn
    )
    print(len(train_loader))  # 看是否为 0
    logger.info("Start train...")
    it=0
    psnrmax=0
    while(it<=ARGS.it_num-1):
        model.train()
        for hdr, crf, invcrf, t in tqdm(train_loader, unit="batch", desc=f"Epoch {it}"):
            
            hdr = hdr.permute(0, 3, 1, 2)
            crf = crf
            invcrf = invcrf
            t = t
            
            # hdr_val, crf_val, invcrf_val, t_val = dataset.read_batch_data()

            # # 转为 Tensor
            # hdr = torch.tensor(hdr_val).float().to(device)
            # hdr = hdr.permute(0, 3, 1, 2)
            # crf = torch.tensor(crf_val).float().to(device)
            # t = torch.tensor(t_val).float().to(device)
            
            loss,lossm, psnr, psnr_no_q = train_step(hdr, crf, t, model, optimizer, device)
            
            if writer:
                writer.add_scalar('Loss/train', loss, it)
                writer.add_scalar('PSNR/train', psnr, it)
                writer.add_scalar('PSNR_NoQ/train', psnr_no_q, it)
            if psnr>psnrmax:
                psnrmax=psnr
                ckpt_path = os.path.join(logdir, f'model_max_{it}.ckpt')
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"Model saved to {ckpt_path}")
                
            logger.info(f"Iter {it} | Loss: {loss:.6f},Lossm:{lossm:.6f}, PSNR: {psnr:.2f}, PSNR No Q: {psnr_no_q:.2f}")
            
            if it<=70000:
                if it % 1000 == 0 or it == ARGS.it_num - 1:
                    ckpt_path = os.path.join(logdir, f'model_{it}.ckpt')
                    torch.save(model.state_dict(), ckpt_path)
                    logger.info(f"Model saved to {ckpt_path}")
            else:
                if it % 100 == 0 or it == ARGS.it_num - 1:
                    ckpt_path = os.path.join(logdir, f'model_{it}.ckpt')
                    torch.save(model.state_dict(), ckpt_path)
                    logger.info(f"Model saved to {ckpt_path}")

            it+=1
        
        

    if writer:
        writer.close()

if __name__ == '__main__':
    main()
    # hdr = torch.rand(8,3,512,512).contiguous()
    # crf = torch.rand(8,1024).contiguous()
    # # invcrf = torch.rand(8,3,512,512).to(device)
    # t = torch.rand(8).contiguous()
    # model = LinearizationNet().to(device)
    # # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # # model.train()
    # # loss, psnr, psnr_no_q = train_step(hdr, crf, t, model, optimizer, device)

    # b, c, h, w = hdr.shape
    # hdr=hdr.permute(0, 2, 3, 1)
    # _hdr_t = hdr * t.view(b, 1, 1, 1)

    # # 添加噪声
    # sigma_s = 0.08 / 6 * torch.rand((b, 1, 1, 3))
    # sigma_c = 0.005 * torch.rand((b, 1, 1, 3))
    # noise_s_map = sigma_s * _hdr_t
    # noise_s = torch.randn_like(_hdr_t) * noise_s_map
    # temp_x = _hdr_t + noise_s
    # noise_c = sigma_c * torch.randn_like(temp_x)
    # temp_x += noise_c
    # _hdr_t = torch.relu(temp_x)

    # clipped_hdr_t = _clip(_hdr_t)
    # ldr = apply_rf(clipped_hdr_t, crf).to(device)
    # out= model(ldr.permute(0, 3, 1, 2))