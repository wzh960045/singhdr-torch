# import logging

# logging.basicConfig(level=logging.INFO)
# import argparse
# import os
# import tensorflow as tf
# from util import get_tensor_shape, apply_rf, get_l2_loss_with_mask
# from dataset import get_train_dataset, RandDatasetReader
# from linearization_net import Linearization_net

# FLAGS = tf.app.flags.FLAGS
# epsilon = 0.001

# # ---

# parser = argparse.ArgumentParser()
# parser.add_argument('--batch_size', type=int, default=8)
# parser.add_argument('--it_num', type=int, default=500000)  # 500k
# parser.add_argument('--logdir_path', type=str, required=True)
# parser.add_argument('--hdr_prefix', type=str, required=True)
# ARGS = parser.parse_args()

# # ---





# # --- graph

# _clip = lambda x: tf.clip_by_value(x, 0, 1)


# def rand_quantize(
#         img,  # [b, h, w, c]
#         is_training,
# ):
#     b, h, w, c, = get_tensor_shape(img)

#     const_bit = tf.constant(8.0, tf.float32, [1, 1, 1, 1])

#     bit = tf.cond(is_training, lambda: const_bit, lambda: const_bit)
#     s = (2 ** bit) - 1

#     img = _clip(img)
#     img = tf.round(s * img) / s
#     img = _clip(img)

#     return img

# def log10(x):
#     numerator = tf.log(x)
#     denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
#     return numerator / denominator


# def build_graph(
#         hdr,  # [b, h, w, c]
#         crf,  # [b, k]
#         invcrf,
#         t,  # [b]
#         is_training,
# ):

#     b, h, w, c, = get_tensor_shape(hdr)
#     b, k, = get_tensor_shape(crf)
#     b, = get_tensor_shape(t)

#     _hdr_t = hdr * tf.reshape(t, [b, 1, 1, 1])

#     # Augment Poisson and Gaussian noise
#     sigma_s = 0.08 / 6 * tf.random_uniform([tf.shape(_hdr_t)[0], 1, 1, 3], minval=0.0, maxval=1.0,
#                                                      dtype=tf.float32, seed=1)
#     sigma_c = 0.005 * tf.random_uniform([tf.shape(_hdr_t)[0], 1, 1, 3], minval=0.0, maxval=1.0, dtype=tf.float32, seed=1)
#     noise_s_map = sigma_s * _hdr_t
#     noise_s = tf.random_normal(shape=tf.shape(_hdr_t), seed=1) * noise_s_map
#     temp_x = _hdr_t + noise_s
#     noise_c = sigma_c * tf.random_normal(shape=tf.shape(_hdr_t), seed=1)
#     temp_x = temp_x + noise_c
#     _hdr_t = tf.nn.relu(temp_x)

#     # Dynamic range clipping
#     clipped_hdr_t = _clip(_hdr_t)

#     # Camera response function
#     ldr = apply_rf(clipped_hdr_t, crf)

#     # Quantization and JPEG compression
#     quantized_hdr = tf.round(ldr * 255.0)
#     quantized_hdr_8bit = tf.cast(quantized_hdr, tf.uint8)
#     jpeg_img_list = []
#     for i in range(ARGS.batch_size):
#         II = quantized_hdr_8bit[i]
#         II = tf.image.adjust_jpeg_quality(II, int(round(float(i)/float(ARGS.batch_size-1)*10.0+90.0)))
#         jpeg_img_list.append(II)
#     jpeg_img = tf.stack(jpeg_img_list, 0)
#     jpeg_img_float = tf.cast(jpeg_img, tf.float32) / 255.0
#     jpeg_img_float.set_shape([None, 256, 256, 3])


#     # loss mask to exclude over-/under-exposed regions
#     gray = tf.image.rgb_to_grayscale(jpeg_img)
#     over_exposed = tf.cast(tf.greater_equal(gray, 249), tf.float32)
#     over_exposed = tf.reduce_sum(over_exposed, axis=[1, 2], keepdims=True)
#     over_exposed = tf.greater(over_exposed, 256.0 * 256.0 * 0.5)
#     under_exposed = tf.cast(tf.less_equal(gray, 6), tf.float32)
#     under_exposed = tf.reduce_sum(under_exposed, axis=[1, 2], keepdims=True)
#     under_exposed = tf.greater(under_exposed, 256.0 * 256.0 * 0.5)
#     extreme_cases = tf.logical_or(over_exposed, under_exposed)
#     loss_mask = tf.cast(tf.logical_not(extreme_cases), tf.float32)

#     lin_net = Linearization_net()
#     pred_invcrf = lin_net.get_output(ldr, is_training)
#     pred_lin_ldr = apply_rf(ldr, pred_invcrf)
#     crf_loss = tf.reduce_mean(tf.square(pred_invcrf - invcrf), axis=1, keepdims=True)
#     loss = get_l2_loss_with_mask(pred_lin_ldr, clipped_hdr_t)

#     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#     with tf.control_dependencies(update_ops):
#         train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(tf.reduce_mean((loss+0.1*crf_loss)*loss_mask))

#     mse = tf.reduce_mean((pred_lin_ldr - clipped_hdr_t) ** 2)
#     psnr = 20.0 * log10(1.0) - 10.0 * log10(mse)
#     mse = tf.reduce_mean((ldr - clipped_hdr_t) ** 2)
#     psnr_no_q = 20.0 * log10(1.0) - 10.0 * log10(mse)

#     tf.summary.scalar('loss', tf.reduce_mean(loss))
#     tf.summary.scalar('crf_loss', tf.reduce_mean(crf_loss))
#     tf.summary.image('pred_lin_ldr', pred_lin_ldr)
#     tf.summary.image('ldr', ldr)
#     tf.summary.image('clipped_hdr_t', clipped_hdr_t)
#     tf.summary.scalar('loss mask 0', tf.squeeze(loss_mask[0]))
#     tf.summary.scalar('loss mask 1', tf.squeeze(loss_mask[1]))
#     tf.summary.scalar('loss mask 2', tf.squeeze(loss_mask[2]))

#     return loss, train_op, psnr, psnr_no_q


# b, h, w, c = ARGS.batch_size, 256, 256, 3

# hdr = tf.placeholder(tf.float32, [None, None, None, c])
# crf = tf.placeholder(tf.float32, [None, None])
# invcrf = tf.placeholder(tf.float32, [None, None])
# t = tf.placeholder(tf.float32, [None])
# is_training = tf.placeholder(tf.bool)

# loss, train_op, psnr, psnr_no_q = build_graph(hdr, crf, invcrf, t, is_training)
# saver = tf.train.Saver(tf.all_variables(), max_to_keep=50)

# # ---

# total_parameters = 0
# for variable in tf.trainable_variables():
#     # shape is an array of tf.Dimension
#     shape = variable.get_shape()
#     print(shape)
#     print(len(shape))
#     variable_parameters = 1
#     for dim in shape:
#         print(dim)
#         variable_parameters *= dim.value
#     print(variable_parameters)
#     total_parameters += variable_parameters
# print('total params: ')
# print(total_parameters)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# lin_net.crf_feature_net.overwrite_init(sess)

# summary = tf.summary.merge_all()
# summary_writer = tf.summary.FileWriter(
#     os.path.join(ARGS.logdir_path, 'summary'),
#     sess.graph,
# )
# dataset_reader = RandDatasetReader(get_train_dataset(ARGS.hdr_prefix), b)

# for it in range(ARGS.it_num):
#     print(it)
#     if it == 0 or it % 10000 == 9999:
#         print('start save')
#         checkpoint_path = os.path.join(ARGS.logdir_path, 'model.ckpt')
#         saver.save(sess, checkpoint_path, global_step=it)
#         print('finish save')
#     hdr_val, crf_val, invcrf_val, t_val = dataset_reader.read_batch_data()
#     _, summary_val = sess.run([train_op, summary], {
#         hdr: hdr_val,
#         crf: crf_val,
#         invcrf: invcrf_val,
#         t: t_val,
#         is_training: True,
#     })
#     if it == 0 or it % 10000 == 9999:
#         summary_writer.add_summary(summary_val, it)
#         logging.info('test')


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
parser.add_argument('--it_num', type=int, default=30000)  # 
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
def train_step(hdr, crf, invcrf,t, model, optimizer, device="cuda"):
    
    

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
    pred_invcrf = model(ldr)
    ldr=ldr.permute(0, 2, 3, 1)
    pred_lin_ldr = apply_rf(ldr.cpu(), pred_invcrf.cpu())
    ldr=ldr.permute(0, 3, 1, 2)
    pred_lin_ldr=pred_lin_ldr.permute(0, 3, 1, 2)
    clipped_hdr_t=clipped_hdr_t.permute(0, 3, 1, 2)
    invcrf=invcrf.to(device)
    crf_loss = (pred_invcrf - invcrf).square().mean(dim=1, keepdim=True)
    loss = get_l2_loss_with_mask(pred_lin_ldr, clipped_hdr_t).to(device)
    
    lossmask = torch.mean((loss+0.1*crf_loss)*loss_mask)
    optimizer.zero_grad()
    lossmask.backward()
    optimizer.step()
    loss=torch.mean(loss)
    psnr = compute_psnr(pred_lin_ldr ,clipped_hdr_t)
    psnr_no_q = compute_psnr(ldr.cpu() , clipped_hdr_t)

    return loss.item(),lossmask.item(), psnr.item(), psnr_no_q.item()

# 主函数
def main():
    logger.info("Start training...")
    
    # 初始化模型
    model=LinearizationNet().to(device)
    model.crf_feature_net.overwrite_init()
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
            
            loss,lossm, psnr, psnr_no_q = train_step(hdr, crf, invcrf,t, model, optimizer, device)
            
            if writer:
                writer.add_scalar('Loss/train', loss, it)
                writer.add_scalar('PSNR/train', psnr, it)
                writer.add_scalar('PSNR_NoQ/train', psnr_no_q, it)
            
            logger.info(f"Iter {it} | Loss: {loss:.4f},Lossm:{lossm:.4f}, PSNR: {psnr:.2f}, PSNR No Q: {psnr_no_q:.2f}")
            if psnr>psnrmax:
                psnrmax=psnr
                ckpt_path = os.path.join(logdir, f'model_max.ckpt')
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"Model saved to {ckpt_path}")
            if it % 10 == 0 or it == ARGS.it_num - 1:
                ckpt_path = os.path.join(logdir, f'model_{it}.ckpt')
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"Model saved to {ckpt_path}")

            it+=1
        
        

    if writer:
        writer.close()

if __name__ == '__main__':
    main()
    
