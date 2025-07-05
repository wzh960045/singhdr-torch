# import logging
# import inspect
# import time

# logging.basicConfig(level=logging.INFO)
# import argparse
# import os
# # import tensorflow as tf
# import hallucination_net
# from util import get_tensor_shape, apply_rf, get_l2_loss
# from dataset import get_train_dataset
# import numpy as np

# # FLAGS = tf.app.flags.FLAGS
# epsilon = 0.001
# # ---

# parser = argparse.ArgumentParser()
# parser.add_argument('--batch_size', type=int, default=8)
# parser.add_argument('--it_num', type=int, default=500000)  # 500k
# parser.add_argument('--logdir_path', type=str, required=True)
# parser.add_argument('--hdr_prefix', type=str, required=True)
# ARGS = parser.parse_args()

# # ---

# # VGG_MEAN = [103.939, 116.779, 123.68]


# # class Vgg16:
# #     def __init__(self, vgg16_npy_path=None):
# #         if vgg16_npy_path is None:
# #             path = inspect.getfile(Vgg16)
# #             path = os.path.abspath(os.path.join(path, os.pardir))
# #             path = os.path.join(path, "vgg16.npy")
# #             vgg16_npy_path = path
# #             print(path)

# #         self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
# #         print("npy file loaded")

# #     def build(self, rgb):
# #         """
# #         load variable from npy to build the VGG
# #         :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
# #         """

# #         start_time = time.time()
# #         print("build model started")
# #         rgb_scaled = rgb * 255.0

# #         # Convert RGB to BGR
# #         red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
# #         bgr = tf.concat(axis=3, values=[
# #             blue - VGG_MEAN[0],
# #             green - VGG_MEAN[1],
# #             red - VGG_MEAN[2],
# #         ])

# #         self.conv1_1 = self.conv_layer(bgr, "conv1_1")
# #         self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
# #         self.pool1 = self.max_pool(self.conv1_2, 'pool1')

# #         self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
# #         self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
# #         self.pool2 = self.max_pool(self.conv2_2, 'pool2')

# #         self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
# #         self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
# #         self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
# #         self.pool3 = self.max_pool(self.conv3_3, 'pool3')

# #         self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
# #         self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
# #         self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
# #         self.pool4 = self.max_pool(self.conv4_3, 'pool4')

# #         self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
# #         self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
# #         self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
# #         self.pool5 = self.max_pool(self.conv5_3, 'pool5')
# #         print(("build model finished: %ds" % (time.time() - start_time)))

# #     def avg_pool(self, bottom, name):
# #         return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

# #     def max_pool(self, bottom, name):
# #         return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

# #     def conv_layer(self, bottom, name):
# #         with tf.variable_scope(name):
# #             filt = self.get_conv_filter(name)

# #             conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

# #             conv_biases = self.get_bias(name)
# #             bias = tf.nn.bias_add(conv, conv_biases)

# #             relu = tf.nn.relu(bias)
# #             return relu

# #     def fc_layer(self, bottom, name):
# #         with tf.variable_scope(name):
# #             shape = bottom.get_shape().as_list()
# #             dim = 1
# #             for d in shape[1:]:
# #                 dim *= d
# #             x = tf.reshape(bottom, [-1, dim])

# #             weights = self.get_fc_weight(name)
# #             biases = self.get_bias(name)

# #             # Fully connected layer. Note that the '+' operation automatically
# #             # broadcasts the biases.
# #             fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

# #             return fc

# #     def get_conv_filter(self, name):
# #         return tf.constant(self.data_dict[name][0], name="filter")

# #     def get_bias(self, name):
# #         return tf.constant(self.data_dict[name][1], name="biases")

# #     def get_fc_weight(self, name):
# #         return tf.constant(self.data_dict[name][0], name="weights")

# import os
# import inspect
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


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

# def get_final(network, x_in):
#     sb, sy, sx, sf = x_in.get_shape().as_list()
#     y_predict = network.outputs

#     # Highlight mask
#     thr = 0.05
#     alpha = tf.reduce_max(x_in, reduction_indices=[3])
#     alpha = tf.minimum(1.0, tf.maximum(0.0, alpha - 1.0 + thr) / thr)
#     alpha = tf.reshape(alpha, [-1, sy, sx, 1])
#     alpha = tf.tile(alpha, [1, 1, 1, 3])

#     # Linearied input and prediction
#     x_lin = tf.pow(x_in, 2.0)
#     y_predict = tf.exp(y_predict) - 1.0 / 255.0

#     # Alpha blending
#     y_final = (1 - alpha) * x_lin + alpha * y_predict

#     return y_final


# def build_graph(
#         hdr,  # [b, h, w, c]
#         crf,  # [b, k]
#         t,  # [b]
#         is_training,
# ):
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

#     # loss mask
#     ldr = apply_rf(clipped_hdr_t, crf)
#     quantized_hdr = tf.round(ldr * 255.0)
#     quantized_hdr_8bit = tf.cast(quantized_hdr, tf.uint8)
#     jpeg_img_list = []
#     for i in range(ARGS.batch_size):
#         II = quantized_hdr_8bit[i]
#         II = tf.image.adjust_jpeg_quality(II, int(round(float(i) / float(ARGS.batch_size - 1) * 10.0 + 90.0)))
#         jpeg_img_list.append(II)
#     jpeg_img = tf.stack(jpeg_img_list, 0)
#     jpeg_img_float = tf.cast(jpeg_img, tf.float32) / 255.0
#     jpeg_img_float.set_shape([None, 256, 256, 3])
#     gray = tf.image.rgb_to_grayscale(jpeg_img)
#     over_exposed = tf.cast(tf.greater_equal(gray, 249), tf.float32)
#     over_exposed = tf.reduce_sum(over_exposed, axis=[1, 2], keepdims=True)
#     over_exposed = tf.greater(over_exposed, 256.0 * 256.0 * 0.5)
#     under_exposed = tf.cast(tf.less_equal(gray, 6), tf.float32)
#     under_exposed = tf.reduce_sum(under_exposed, axis=[1, 2], keepdims=True)
#     under_exposed = tf.greater(under_exposed, 256.0 * 256.0 * 0.5)
#     extreme_cases = tf.logical_or(over_exposed, under_exposed)
#     loss_mask = tf.cast(tf.logical_not(extreme_cases), tf.float32)

#     # Highlight mask
#     thr = 0.12
#     alpha = tf.reduce_max(clipped_hdr_t, reduction_indices=[3])
#     alpha = tf.minimum(1.0, tf.maximum(0.0, alpha - 1.0 + thr) / thr)
#     alpha = tf.reshape(alpha, [-1, tf.shape(clipped_hdr_t)[1], tf.shape(clipped_hdr_t)[2], 1])
#     alpha = tf.tile(alpha, [1, 1, 1, 3])

#     with tf.variable_scope("Hallucination_Net"):
#         net, vgg16_conv_layers = hallucination_net.model(clipped_hdr_t, ARGS.batch_size, True)
#         y_predict = tf.nn.relu(net.outputs)
#         y_final = (clipped_hdr_t) + alpha * y_predict # residual

#     with tf.variable_scope("Hallucination_Net", reuse=True):
#         net_test, vgg16_conv_layers_test = hallucination_net.model(clipped_hdr_t, ARGS.batch_size, False)
#         y_predict_test = tf.nn.relu(net_test.outputs)
#         y_final_test = (clipped_hdr_t) + alpha * y_predict_test # residual


#     # _log = lambda x: tf.log(x + 1.0/255.0)

#     vgg = Vgg16('vgg16.npy')
#     vgg.build(tf.log(1.0+10.0*y_final)/tf.log(1.0+10.0))
#     vgg2 = Vgg16('vgg16.npy')
#     vgg2.build(tf.log(1.0+10.0*_hdr_t)/tf.log(1.0+10.0))
#     perceptual_loss = tf.reduce_mean(tf.abs((vgg.pool1 - vgg2.pool1)), axis=[1, 2, 3], keepdims=True)
#     perceptual_loss += tf.reduce_mean(tf.abs((vgg.pool2 - vgg2.pool2)), axis=[1, 2, 3], keepdims=True)
#     perceptual_loss += tf.reduce_mean(tf.abs((vgg.pool3 - vgg2.pool3)), axis=[1, 2, 3], keepdims=True)

#     loss_test = get_l2_loss(_log(y_final_test), _log(_hdr_t))

#     y_final_gamma = tf.log(1.0+10.0*y_final)/tf.log(1.0+10.0)
#     _hdr_t_gamma = tf.log(1.0+10.0*_hdr_t)/tf.log(1.0+10.0)

#     loss = tf.reduce_mean(tf.abs(y_final_gamma - _hdr_t_gamma), axis=[1, 2, 3], keepdims=True)
#     y_final_gamma_pad_x = tf.pad(y_final_gamma, [[0, 0], [0, 1], [0, 0], [0, 0]], 'SYMMETRIC')
#     y_final_gamma_pad_y = tf.pad(y_final_gamma, [[0, 0], [0, 0], [0, 1], [0, 0]], 'SYMMETRIC')
#     tv_loss_x = tf.reduce_mean(tf.abs(y_final_gamma_pad_x[:, 1:] - y_final_gamma_pad_x[:, :-1]))
#     tv_loss_y = tf.reduce_mean(tf.abs(y_final_gamma_pad_y[:, :, 1:] - y_final_gamma_pad_y[:, :, :-1]))
#     tv_loss = tv_loss_x + tv_loss_y

#     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#     with tf.control_dependencies(update_ops):
#         train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(
#             tf.reduce_mean((loss + 0.001 * perceptual_loss + 0.1 * tv_loss)*loss_mask))

#     t_vars = tf.trainable_variables()
#     print('all layers:')
#     for var in t_vars: print(var.name)

#     tf.summary.scalar('loss', tf.reduce_mean(loss))
#     tf.summary.image('hdr_t', _hdr_t)
#     tf.summary.image('y', y_final)
#     tf.summary.image('clipped_hdr_t', clipped_hdr_t)
#     tf.summary.image('alpha', alpha)
#     tf.summary.image('y_predict', y_predict)
#     tf.summary.image('log_hdr_t', tf.log(1.0+10.0*_hdr_t)/tf.log(1.0+10.0))
#     tf.summary.image('log_y', tf.log(1.0+10.0*y_final)/tf.log(1.0+10.0))
#     tf.summary.image('log_clipped_hdr_t', tf.log(1.0+10.0*clipped_hdr_t)/tf.log(1.0+10.0))


#     psnr = tf.zeros([])
#     psnr_no_q = tf.zeros([])

#     return loss, train_op, psnr, psnr_no_q, loss_test, vgg16_conv_layers, net, y_final_test, y_predict_test, alpha


# b, h, w, c = ARGS.batch_size, 512, 512, 3

# hdr = tf.placeholder(tf.float32, [None, None, None, c])
# crf = tf.placeholder(tf.float32, [None, None])
# t = tf.placeholder(tf.float32, [None])
# is_training = tf.placeholder(tf.bool)

# loss, train_op, psnr, psnr_no_q, loss_test, vgg16_conv_layers, net, y_final_test, y_predict_test, alpha = build_graph(hdr, crf, t, is_training)
# saver = tf.train.Saver(tf.all_variables(), max_to_keep=50)

# # ---

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# hallucination_net.load_vgg_weights(vgg16_conv_layers, 'vgg16_places365_weights.npy', sess)

# summary = tf.summary.merge_all()
# summary_writer = tf.summary.FileWriter(
#     os.path.join(ARGS.logdir_path, 'summary'),
#     sess.graph,
# # )
# # dataset_reader = RandDatasetReader(get_train_dataset(ARGS.hdr_prefix), b)

# for it in range(ARGS.it_num):
#     print(it)
#     if it == 0 or it % 10000 == 9999:
#         print('start save')
#         checkpoint_path = os.path.join(ARGS.logdir_path, 'model.ckpt')
#         saver.save(sess, checkpoint_path, global_step=it)
#         print(net.all_params)
#         # tl.files.save_npz(net.all_params, name=os.path.join(ARGS.logdir_path, 'model'+str(it)+'.npz'), sess=sess)
#         print('finish save')
#     hdr_val, crf_val, invcrf_val, t_val = dataset_reader.read_batch_data()
#     _, summary_val = sess.run([train_op, summary], {
#         hdr: hdr_val,
#         crf: crf_val,
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
import torch.nn.functional as F
from util import get_tensor_shape, apply_rf, get_l2_loss_with_mask
from dataset import get_train_dataset,hdr_collate_fn
from model.team02_dequantization_net import Dequantization_net
from model.team02_hallucination_net import HDRAutoencoder
from model.team02_linearization_net import LinearizationNet
import inspect
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--it_num', type=int, default=100000)  # 
parser.add_argument('--logdir_path', type=str, required=True)
parser.add_argument('--hdr_prefix', type=str, required=True)
parser.add_argument('--resume', type=str, default=None, help="Path to model checkpoint")
ARGS = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
epsilon = 0.001
__all__ = ['Vgg16']

VGG_MEAN = [103.939, 116.779, 123.68]  # BGR mean for VGG


class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print(path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1', allow_pickle=True).item()
        # print("npy file loaded")
        # rgb = torch.rand(1,3,512,512).contiguous().to(device)
        # self.build(rgb)  # 初始化结构（输入为 None）
        # self.load_weights()

    def build(self, rgb):
        """
        Load variable from npy to build the VGG
        :param rgb: Tensor of shape [batch, 3, height, width], scaled [0, 1]
        """
        self.layers = {}
        start_time = self._now()

        # print("build model started")
        # print(rgb.shape)
        # Convert RGB to BGR and preprocess
        rgb_scaled = rgb * 255.0
        red, green, blue = rgb_scaled.split(1, dim=1)
        bgr = torch.cat([
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ], dim=1)  # (B, C, H, W), C=3
        # print(bgr.shape)
        # Layer building
        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')
        print(f"build model finished: {self._now() - start_time:.2f}s")

    def avg_pool(self, bottom, name):
        x = F.avg_pool2d(bottom, kernel_size=2, stride=2, padding=0)
        self.layers[name] = x
        return x

    def max_pool(self, bottom, name):
        x = F.max_pool2d(bottom, kernel_size=2, stride=2, padding=0)
        self.layers[name] = x
        return x

    def conv_layer(self, bottom, name):
        with torch.no_grad():
            filt = self.get_conv_filter(name)
            conv_biases = self.get_bias(name)
            conv = F.conv2d(bottom, filt, bias=conv_biases, stride=1, padding=1)
            # 
            # bias = torch.add(conv, conv_biases)
            relu = F.relu(conv)
            self.layers[name] = relu
            return relu

    def fc_layer(self, bottom, name):
        with torch.no_grad():
            shape = bottom.shape
            dim = np.prod(shape[1:])
            x = bottom.view(-1, int(dim))

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            fc = torch.add(torch.matmul(x, weights), biases)
            self.layers[name] = fc
            return fc

    def get_conv_filter(self, name):
        data = self.data_dict[name][0]
        # TF: (H, W, Cin, Cout)
        # PyTorch: (Cout, Cin, H, W)
        data_torch = torch.from_numpy(data).permute(3, 2, 0, 1).float().to(device)
        return data_torch

    def get_bias(self, name):
        data = self.data_dict[name][1]
        data_torch = torch.from_numpy(data).float().to(device)
        return data_torch

    def get_fc_weight(self, name):
        data = self.data_dict[name][0]
        data_torch = torch.from_numpy(data).float().to(device)
        return data_torch

    def load_weights(self, ignore_missing=False):
        pass  # 权重已直接加载到层中

    def _now(self):
        import time
        return time.time()

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
    sigma_s = 0.08 / 6 * torch.rand((b, 1, 1, 3)).to(device)
    sigma_c = 0.005 * torch.rand((b, 1, 1, 3)).to(device)
    noise_s_map = sigma_s * _hdr_t
    noise_s = torch.randn_like(_hdr_t) * noise_s_map
    temp_x = _hdr_t + noise_s
    noise_c = sigma_c * torch.randn_like(temp_x)
    temp_x += noise_c
    _hdr_t = torch.relu(temp_x)

    clipped_hdr_t = _clip(_hdr_t)

    # 应用 CRF
    ldr = apply_rf(clipped_hdr_t.cpu(), crf.cpu())
    ldr=ldr.permute(0, 3, 1, 2).to(device)

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

    thr = 0.12
    # Step 1: 取每个像素点 RGB 三个通道的最大值 (H, W)
    alpha = torch.max(clipped_hdr_t, dim=3, keepdim=False)[0]  # shape: [B, H, W]
    # Step 2: 计算 alpha mask（范围 [0, 1]）
    alpha = torch.clamp((alpha - 1.0 + thr) / thr, min=0.0, max=1.0)
    # Step 3: 扩展维度到 [B, H, W, 1]
    alpha = alpha.unsqueeze(-1)  # shape: [B, H, W, 1]
    # Step 4: 扩展到 3 个通道，得到 [B, H, W, 3]
    alpha = alpha.expand(-1, -1, -1, 3).permute(0, 3, 1, 2).to(device)  # 或者使用 repeat
    clipped_hdr_t=clipped_hdr_t.permute(0, 3, 1, 2)
    net, vgg16_conv_layers = model(clipped_hdr_t)
    y_predict = F.relu(net)
    y_final = (clipped_hdr_t) + alpha * y_predict 
    
    _hdr_t=_hdr_t.permute(0, 3, 1, 2)
    # 假设 y_final 和 _hdr_t 是已有的 PyTorch 张量
    # 计算输入到 VGG 网络的变换
    input_vgg_1 = torch.log(1.0 + 10.0 * y_final) / torch.log(torch.tensor(1.0 + 10.0))
    input_vgg_2 = torch.log(1.0 + 10.0 * _hdr_t) / torch.log(torch.tensor(1.0 + 10.0))

    # 构建 VGG 模型实例并传入数据
    vgg = Vgg16('vgg16.npy')
    vgg.build(input_vgg_1)
    vgg2 = Vgg16('vgg16.npy')
    vgg2.build(input_vgg_2)

    # 计算感知损失
    perceptual_loss = torch.abs(vgg.pool1-vgg2.pool1).mean(dim=[1, 2, 3], keepdim=True)
    perceptual_loss += torch.abs(vgg.pool2-vgg2.pool2).mean(dim=[1, 2, 3], keepdim=True)
    perceptual_loss += torch.abs(vgg.pool3-vgg2.pool3).mean(dim=[1, 2, 3], keepdim=True)

    
    y_final_gamma = torch.log(1.0 + 10.0 * y_final) /torch.log(torch.tensor(1.0 + 10.0)) 
    _hdr_t_gamma = torch.log(1.0 + 10.0 * _hdr_t) / torch.log(torch.tensor(1.0 + 10.0))

    # Step 2: L1 loss between transformed outputs
    loss = torch.abs(y_final_gamma - _hdr_t_gamma).mean(dim=[1, 2, 3], keepdim=True)

    # Step 3: TV Loss（Total Variation Regularization）

    # x方向差分（在H维度做padding和差分）
    y_final_pad_x = F.pad(y_final_gamma, (0, 0, 0, 1, 0, 0), mode='reflect')  # pad bottom
    tv_loss_x = torch.mean(torch.abs(y_final_pad_x[:, :, 1:, :] - y_final_pad_x[:, :, :-1, :]))

    # y方向差分（在W维度做padding和差分）
    y_final_pad_y = F.pad(y_final_gamma, (0, 0, 0, 0, 0, 1), mode='reflect')  # pad right
    tv_loss_y = torch.mean(torch.abs(y_final_pad_y[:, :, :, 1:] - y_final_pad_y[:, :, :, :-1]))

    # 合并 TV Loss
    tv_loss = tv_loss_x + tv_loss_y

    
    lossmask = torch.mean((loss + 0.001 * perceptual_loss + 0.1 * tv_loss)*loss_mask)

    optimizer.zero_grad()
    lossmask.backward()
    optimizer.step()
    loss=torch.mean(loss)
    psnr = torch.tensor(0)
    psnr_no_q = torch.tensor(0)

    return loss.item(),lossmask.item(), psnr.item(), psnr_no_q.item()

# 主函数
def main():
    logger.info("Start training...")
    
    # 初始化模型
    model = HDRAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    load_vgg_weights(model.encoder,'vgg16_places365_weights.npy')
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
    
    lossmin=100
    while(it<=ARGS.it_num-1):
        model.train()
        for hdr, crf, invcrf, t in tqdm(train_loader, unit="batch", desc=f"iter {it}"):
            
            hdr = hdr.permute(0, 3, 1, 2).to(device)
            crf = crf.to(device)
            invcrf = invcrf.to(device)
            t = t.to(device)
            
            
            loss,lossm, psnr, psnr_no_q = train_step(hdr, crf, t, model, optimizer, device)
            
            if writer:
                writer.add_scalar('Loss/train', loss, it)
                writer.add_scalar('PSNR/train', psnr, it)
                writer.add_scalar('PSNR_NoQ/train', psnr_no_q, it)

            if lossm<lossmin:
                lossmin=lossm
                ckpt_path = os.path.join(logdir, f'model_best_{it}.ckpt')
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"Model saved to {ckpt_path}")

            logger.info(f"Iter {it} | Loss: {loss:.4f},Lossm:{lossm:.4f}")

            
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
    # hdr = torch.rand(2,3,512,512).contiguous()
    # vgg = Vgg16('vgg16.npy')
    # vgg.build(hdr)
    