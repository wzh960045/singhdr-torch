# from net import BaseNet, AggNet
# # import tensorflow.contrib.slim as slim
# # import tensorflow as tf
import numpy as np
import os
from util import get_tensor_shape

# class CrfFeatureNettf(BaseNet):
#     def __init__(
#             self,
#             scope='crf_feature_net',
#     ):
#         super().__init__(scope)
#         return

#     def conv(
#             self,
#             input,
#             k_h,
#             k_w,
#             c_o,
#             s_h,
#             s_w,
#             name,
#             relu=True,
#             padding='SAME',
#             biased=True
#     ):
#         assert s_h == s_w
#         output = slim.conv2d(
#             input,
#             c_o,
#             [k_h, k_w],
#             stride=s_h,
#             scope=name,
#             activation_fn=(tf.nn.relu if relu else None),
#             padding=padding,
#             biases_initializer=(tf.zeros_initializer() if biased else None),
#         )
#         return output

#     def batch_normalization(
#             self,
#             input,
#             is_training,
#             name,
#             relu=False,
#     ):
#         with tf.variable_scope(name) as scope:
#             output = slim.batch_norm(
#                 input,
#                 scale=True,
#                 activation_fn=(tf.nn.relu if relu else None),
#                 is_training=is_training,
#             )
#         return output

#     def max_pool(
#             self,
#             input,
#             k_h,
#             k_w,
#             s_h,
#             s_w,
#             name,
#             padding='SAME',
#     ):
#         output = tf.nn.max_pool(
#             input,
#             ksize=[1, k_h, k_w, 1],
#             strides=[1, s_h, s_w, 1],
#             padding=padding,
#             name=name,
#         )
#         return output

#     def avg_pool(
#             self,
#             input,
#             k_h,
#             k_w,
#             s_h,
#             s_w,
#             name,
#             padding='SAME',
#     ):
#         output = tf.nn.avg_pool(
#             input,
#             ksize=[1, k_h, k_w, 1],
#             strides=[1, s_h, s_w, 1],
#             padding=padding,
#             name=name,
#         )
#         return output

#     def fc(
#             self,
#             input,
#             num_out,
#             name,
#             relu=True,
#     ):
#         output = slim.fully_connected(
#             input,
#             num_out,
#             activation_fn=(tf.nn.relu if relu else None),
#             scope=name,
#         )
#         return output

#     def _get_output(
#             self,
#             ldr,  # [b, 227, 227, c]
#             is_training,
#     ):

#         conv1 = self.conv(ldr, 7, 7, 64, 2, 2, relu=False, name='conv1')
#         bn_conv1 = self.batch_normalization(conv1, is_training, relu=True, name='bn_conv1')
#         pool1 = self.max_pool(bn_conv1, 3, 3, 2, 2, name='pool1')
#         res2a_branch1 = self.conv(pool1, 1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
#         bn2a_branch1 = self.batch_normalization(res2a_branch1, is_training, name='bn2a_branch1')

#         res2a_branch2a = self.conv(pool1, 1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
#         bn2a_branch2a = self.batch_normalization(res2a_branch2a, is_training, relu=True, name='bn2a_branch2a')
#         res2a_branch2b = self.conv(bn2a_branch2a, 3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
#         bn2a_branch2b = self.batch_normalization(res2a_branch2b, is_training, relu=True, name='bn2a_branch2b')
#         res2a_branch2c = self.conv(bn2a_branch2b, 1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
#         bn2a_branch2c = self.batch_normalization(res2a_branch2c, is_training, name='bn2a_branch2c')

#         res2a_relu = tf.nn.relu(bn2a_branch1 + bn2a_branch2c)
#         res2b_branch2a = self.conv(res2a_relu, 1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
#         bn2b_branch2a = self.batch_normalization(res2b_branch2a, is_training, relu=True, name='bn2b_branch2a')
#         res2b_branch2b = self.conv(bn2b_branch2a, 3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
#         bn2b_branch2b = self.batch_normalization(res2b_branch2b, is_training, relu=True, name='bn2b_branch2b')
#         res2b_branch2c = self.conv(bn2b_branch2b, 1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c')
#         bn2b_branch2c = self.batch_normalization(res2b_branch2c, is_training, name='bn2b_branch2c')

#         res2b_relu = tf.nn.relu(res2a_relu + bn2b_branch2c)
#         res2c_branch2a = self.conv(res2b_relu, 1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
#         bn2c_branch2a = self.batch_normalization(res2c_branch2a, is_training, relu=True, name='bn2c_branch2a')
#         res2c_branch2b = self.conv(bn2c_branch2a, 3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
#         bn2c_branch2b = self.batch_normalization(res2c_branch2b, is_training, relu=True, name='bn2c_branch2b')
#         res2c_branch2c = self.conv(bn2c_branch2b, 1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c')
#         bn2c_branch2c = self.batch_normalization(res2c_branch2c, is_training, name='bn2c_branch2c')

#         res2c_relu = tf.nn.relu(res2b_relu + bn2c_branch2c)
#         res3a_branch1 = self.conv(res2c_relu, 1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1')
#         bn3a_branch1 = self.batch_normalization(res3a_branch1, is_training, name='bn3a_branch1')

#         res3a_branch2a = self.conv(res2c_relu, 1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a')
#         bn3a_branch2a = self.batch_normalization(res3a_branch2a, is_training, relu=True, name='bn3a_branch2a')
#         res3a_branch2b = self.conv(bn3a_branch2a, 3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
#         bn3a_branch2b = self.batch_normalization(res3a_branch2b, is_training, relu=True, name='bn3a_branch2b')
#         res3a_branch2c = self.conv(bn3a_branch2b, 1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c')
#         bn3a_branch2c = self.batch_normalization(res3a_branch2c, is_training, name='bn3a_branch2c')

#         res3a_relu = tf.nn.relu(bn3a_branch1 + bn3a_branch2c)
#         res3b_branch2a = self.conv(res3a_relu, 1, 1, 128, 1, 1, biased=False, relu=False, name='res3b_branch2a')
#         bn3b_branch2a = self.batch_normalization(res3b_branch2a, is_training, relu=True, name='bn3b_branch2a')
#         res3b_branch2b = self.conv(bn3b_branch2a, 3, 3, 128, 1, 1, biased=False, relu=False, name='res3b_branch2b')
#         bn3b_branch2b = self.batch_normalization(res3b_branch2b, is_training, relu=True, name='bn3b_branch2b')
#         res3b_branch2c = self.conv(bn3b_branch2b, 1, 1, 512, 1, 1, biased=False, relu=False, name='res3b_branch2c')
#         bn3b_branch2c = self.batch_normalization(res3b_branch2c, is_training, name='bn3b_branch2c')

#         res3b_relu = tf.nn.relu(res3a_relu + bn3b_branch2c)
#         pool5 = self.avg_pool(res3b_relu, 7, 7, 1, 1, padding='VALID', name='pool5')
#         # fc11      = self.fc(tf.contrib.layers.flatten(pool5), 11, relu=False, name='fc11')

#         return tf.reduce_mean(res3b_relu, [1, 2], keep_dims=False)

#     def overwrite_init(self, sess):

#         # np_var

#         def refine_np_var(input, output_dict, curr_tag=''):
#             if type(input) is dict:
#                 for key, val in input.items():
#                     if 'fc11' not in key:
#                         refine_np_var(val, output_dict, curr_tag + '/%s' % key)
#             else:
#                 assert curr_tag not in output_dict
#                 output_dict[curr_tag] = input

#         np_var = {}
#         refine_np_var(
#             np.load('crf_net_v2.npy', encoding='latin1').item(),
#             np_var,
#         )

#         # tf_var

#         def tf_name_2_np_name(tf_name):
#             np_name = tf_name
#             np_name = np_name.replace(':0', '')
#             np_name = np_name.replace('/BatchNorm', '')
#             np_name = np_name.replace('%s' % self.scope, '')
#             '''
#             offset = beta
#             scale = gamma
#             '''
#             np_name = np_name.replace('beta', 'offset')
#             np_name = np_name.replace('gamma', 'scale')
#             np_name = np_name.replace('moving_variance', 'variance')
#             np_name = np_name.replace('moving_mean', 'mean')
#             return np_name

#         tf_var = {tf_name_2_np_name(var.name): var for var in tf.get_collection(
#             tf.GraphKeys.GLOBAL_VARIABLES,
#             scope=self.scope,
#         )}

#         # chk all
#         print(tf_var)

#         for key, var in np_var.items():
#             print(key)
#             assert key in tf_var

#         # load all

#         for key, var in np_var.items():
#             if '/conv1/' not in key:
#                 tf_var[key].load(var, sess)

#         return


# class AEInvcrfDecodeNettf(BaseNet):

#     def __init__(
#             self,
#             n_digit=2,
#     ):
#         super().__init__('ae_invcrf_decode_net')

#         self.n_digit = n_digit
#         self.decode_spec = []
#         self.s = 1024
#         self.n_p = 12
#         self.act = tf.nn.tanh
#         self.reg = tf.contrib.layers.l2_regularizer(1e-3)

#         return

#     def _f(
#             self,
#             p,  # [b, n_p]
#     ):
#         '''
#         m =
#         x_0^1, x_1^1
#         x_0^2, x_1^2
#         '''
#         m = []
#         for i in range(self.n_p):
#             m.append([x ** (i + 1) for x in np.linspace(0, 1, num=self.s, dtype='float64')])
#         m = tf.constant(m, dtype=tf.float64)  # [n_c, s]
#         return tf.matmul(
#             p,  # [b, n_p]
#             m,  # [n_p, s]
#         )  # [b, s]

#     def _decode(
#             self,
#             x,  # [b, n_digit]
#     ):
#         def parse_dorf():

#             with open(os.path.join('dorfCurves.txt'), 'r') as f:
#                 lines = f.readlines()
#                 lines = [line.strip() for line in lines]

#             i = [lines[idx + 3] for idx in range(0, len(lines), 6)]
#             b = [lines[idx + 5] for idx in range(0, len(lines), 6)]

#             i = [ele.split() for ele in i]
#             b = [ele.split() for ele in b]

#             i = np.float32(i)
#             b = np.float32(b)

#             return i, b

#         def _parse(lines, tag):

#             for line_idx, line in enumerate(lines):
#                 if line == tag:
#                     break

#             s_idx = line_idx + 1

#             r = []
#             for idx in range(s_idx, s_idx + int(1024 / 4)):
#                 r += lines[idx].split()

#             return np.float32(r)

#         # e, f0, h
#         # [1024], [1024], [1024, 11]
#         def parse_emor():

#             with open(os.path.join('emor.txt'), 'r') as f:
#                 lines = f.readlines()
#                 lines = [line.strip() for line in lines]

#             e = _parse(lines, 'E =')
#             f0 = _parse(lines, 'f0 =')
#             h = np.stack([_parse(lines, 'h(%d)=' % (i + 1)) for i in range(11)], axis=-1)

#             return e, f0, h

#         # b, g0, hinv
#         # [1024], [1024], [1024, 11]
#         def parse_invemor():

#             with open(os.path.join('invemor.txt'), 'r') as f:
#                 lines = f.readlines()
#                 lines = [line.strip() for line in lines]

#             b = _parse(lines, 'B =')
#             g0 = _parse(lines, 'g0 =')
#             hinv = np.stack([_parse(lines, 'hinv(%d)=' % (i + 1)) for i in range(11)], axis=-1)

#             return b, g0, hinv

#         # _, B = parse_dorf()
#         # _, F0, H = parse_emor()

#         def invcrf_pca_w_2_invcrf(
#                 invcrf_pca_w,  # [b, 11]
#         ):
#             _, G0, HINV = parse_invemor()
#             b, _, = get_tensor_shape(invcrf_pca_w)

#             invcrf_pca_w = tf.expand_dims(invcrf_pca_w, -1)  # [b, 11, 1]

#             G0 = tf.constant(G0)  # [   s   ]
#             G0 = tf.reshape(G0, [1, -1, 1])  # [1, s, 1]

#             HINV = tf.constant(HINV)  # [   s, 11]
#             HINV = tf.expand_dims(HINV, 0)  # [1, s, 11]
#             HINV = tf.tile(HINV, [b, 1, 1])  # [b, s, 11]

#             invcrf = G0 + tf.matmul(
#                 HINV,  # [b, s, 11]
#                 invcrf_pca_w,  # [b, 11, 1]
#             )  # [b, s, 1]

#             invcrf = tf.squeeze(invcrf, -1)  # [b, s]

#             return invcrf

#         for c in self.decode_spec:
#             x = tf.layers.dense(x, c, activation=self.act, kernel_regularizer=self.reg)
#         x = tf.layers.dense(x, self.n_p - 1)  # [b, n_p - 1]
#         invcrf = invcrf_pca_w_2_invcrf(x)
#         # x = tf.concat([x, 1.0 - tf.reduce_sum(x, axis=-1, keep_dims=True)], -1) # [b, n_p]
#         # x = self._f(x) # [b, s]
#         return invcrf

#     # [b, s]
#     def _get_output(
#             self,
#             feature,  # [b, n_digit]
#     ):
#         return self._decode(feature)


# class Linearization_nettf(AggNet):

#     def __init__(self):
#         self.crf_feature_net = CrfFeatureNettf()
#         self.ae_invcrf_decode_net = AEInvcrfDecodeNettf()
#         super().__init__([
#             self.crf_feature_net,
#             self.ae_invcrf_decode_net,
#         ])
#         return

#     @staticmethod
#     def _resize_img(img, t):
#         _, h, w, _, = get_tensor_shape(img)
#         ratio = h / w
#         pred = tf.greater(ratio, 1.0)
#         _round = lambda x: tf.cast(tf.round(x), tf.int32)
#         t_h = tf.cond(
#             pred,
#             lambda: _round(t * ratio),
#             lambda: t,
#         )
#         t_w = tf.cond(
#             pred,
#             lambda: t,
#             lambda: _round(t / ratio),
#         )
#         img = tf.image.resize_images(
#             img,
#             [t_h, t_w],
#             method=tf.image.ResizeMethod.BILINEAR,
#         )
#         img = tf.image.resize_image_with_crop_or_pad(img, t, t)
#         return img

#     @staticmethod
#     def _increase(rf):
#         g = rf[:, 1:] - rf[:, :-1]
#         # [b, 1023]

#         min_g = tf.reduce_min(g, axis=-1, keep_dims=True)
#         # [b, 1]

#         # r = tf.nn.relu(1e-6 - min_g)
#         r = tf.nn.relu(-min_g)
#         # [b, 1023]

#         new_g = g + r
#         # [b, 1023]

#         new_g = new_g / tf.reduce_sum(new_g, axis=-1, keep_dims=True)
#         # [b, 1023]

#         new_rf = tf.cumsum(new_g, axis=-1)
#         # [b, 1023]

#         new_rf = tf.pad(new_rf, [[0, 0], [1, 0]], 'CONSTANT')
#         # [b, 1024]

#         return new_rf

#     def _get_output(self, img, is_training):
#         # edge branch

#         edge_1 = tf.image.sobel_edges(img)

#         edge_1 = tf.reshape(edge_1, [tf.shape(img)[0], tf.shape(img)[1], tf.shape(img)[2], 6])

#         tf.summary.image('edge0', edge_1[:, :, :, 0:3])
#         tf.summary.image('edge1', edge_1[:, :, :, 3:6])

#         # edge_1 = tf.reshape(edge_1, [-1, img.get_shape().as_list()[1], img.get_shape().as_list()[2], 1])

#         def histogram_layer(img, max_bin):
#             # histogram branch
#             tmp_list = []
#             for i in range(max_bin + 1):
#                 histo = tf.nn.relu(1 - tf.abs(img - i / float(max_bin)) * float(max_bin))
#                 tmp_list.append(histo)
#             histogram_tensor = tf.concat(tmp_list, -1)
#             return histogram_tensor
#             # histogram_tensor = tf.layers.average_pooling2d(histogram_tensor, 16, 1, 'same')

#         feature = self.crf_feature_net.get_output(
#             tf.concat([img, edge_1, histogram_layer(img, 4), histogram_layer(img, 8), histogram_layer(img, 16)], -1),
#             is_training)
#         feature = tf.cast(feature, tf.float32)

#         invcrf = self.ae_invcrf_decode_net.get_output(feature)
#         # [b, 1024]

#         invcrf = self._increase(invcrf)
#         # [b, 1024]

#         invcrf = tf.cast(invcrf, tf.float32)
#         # float32

#         return invcrf
    









import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class CrfFeatureNet(nn.Module):
    def __init__(self):#, scope='crf_feature_net'
        # super().__init__()#scope
        super(CrfFeatureNet, self).__init__()
        self.scope='crf_feature_net'
        self.conv1 = nn.Conv2d((3 + 6 + 15 + 27 + 51), 64,kernel_size=7, stride=2, padding=3)
        self.bn_conv1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Res2a branches
        self.res2a_branch1 =nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
 
        self.bn2a_branch1=nn.BatchNorm2d(256)
        self.res2a_branch2a = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False)
        
        self.bn2a_branch2a =nn.BatchNorm2d(64)
        self.res2a_branch2b =  nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn2a_branch2b =nn.BatchNorm2d(64)
        self.res2a_branch2c = nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
        
        self.bn2a_branch2c = nn.BatchNorm2d(256)
        # Res2b
        self.res2b_branch2a = nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False)
        
        self.bn2b_branch2a=nn.BatchNorm2d(64)
        self.res2b_branch2b =  nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn2b_branch2b =nn.BatchNorm2d(64)
        self.res2b_branch2c = nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
        
        self.bn2b_branch2c =nn.BatchNorm2d(256)

        # Res2c
        self.res2c_branch2a = nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False)
            
        
        self.bn2c_branch2a = nn.BatchNorm2d(64)
        self.res2c_branch2b =  nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn2c_branch2b =nn.BatchNorm2d(64) 
        self.res2c_branch2c = nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
       
        self.bn2c_branch2c = nn.BatchNorm2d(256)
        # Res3a
        self.res3a_branch1 =nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False)
            
        
        self.bn3a_branch1 = nn.BatchNorm2d(512)
        self.res3a_branch2a =  nn.Conv2d(256, 128, kernel_size=1, stride=2, bias=False)
        
        self.bn3a_branch2a = nn.BatchNorm2d(128)
        self.res3a_branch2b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn3a_branch2b = nn.BatchNorm2d(128)
        self.res3a_branch2c = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        
        self.bn3a_branch2c = nn.BatchNorm2d(512)
        # Res3b
        self.res3b_branch2a =  nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False)
            
        
        self.bn3b_branch2a = nn.BatchNorm2d(128)
        self.res3b_branch2b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
            
        
        self.bn3b_branch2b =nn.BatchNorm2d(128)
        self.res3b_branch2c = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
            
        
        self.bn3b_branch2c=nn.BatchNorm2d(512)

        # Final pooling
        self.pool5 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, ldr, is_training=True):
        """
        Args:
            ldr: input tensor of shape [B, C, H, W] (NCHW format)
        Returns:
            output tensor after feature extraction
        """
        # Initial layers
        x = ldr
        conv1 = self.conv1(x)
        bn_conv1 = self.bn_conv1(conv1)
        x = F.relu(bn_conv1)
        pool1 = self.pool1(x)
        
        # Res2a
        res2a_branch1_out = self.bn2a_branch1(self.res2a_branch1(pool1))
        res2a_branch2a_out = self.bn2a_branch2a(self.res2a_branch2a(pool1))
        res2a_branch2a_relu = F.relu(res2a_branch2a_out)
        res2a_branch2b_out = self.bn2a_branch2b(self.res2a_branch2b(res2a_branch2a_relu))
        res2a_branch2b_relu = F.relu(res2a_branch2b_out)
        res2a_branch2c_out = self.bn2a_branch2c(self.res2a_branch2c(res2a_branch2b_relu))
        res2a_out = F.relu(res2a_branch1_out + res2a_branch2c_out)

        # Res2b
        res2b_branch2a_out = self.bn2b_branch2a(self.res2b_branch2a(res2a_out))
        res2b_branch2a_relu = F.relu(res2b_branch2a_out)
        res2b_branch2b_out = self.bn2b_branch2b(self.res2b_branch2b(res2b_branch2a_relu))
        res2b_branch2b_relu = F.relu(res2b_branch2b_out)
        res2b_branch2c_out = self.bn2b_branch2c(self.res2b_branch2c(res2b_branch2b_relu))
        res2b_out = F.relu(res2a_out + res2b_branch2c_out)

        # Res2c
        res2c_branch2a_out = self.bn2c_branch2a(self.res2c_branch2a(res2b_out))
        res2c_branch2a_relu = F.relu(res2c_branch2a_out)
        res2c_branch2b_out = self.bn2c_branch2b(self.res2c_branch2b(res2c_branch2a_relu))
        res2c_branch2b_relu = F.relu(res2c_branch2b_out)
        res2c_branch2c_out = self.bn2c_branch2c(self.res2c_branch2c(res2c_branch2b_relu))
        res2c_out = F.relu(res2b_out + res2c_branch2c_out)

        # Res3a
        res3a_branch1_out = self.bn3a_branch1(self.res3a_branch1(res2c_out))
        res3a_branch2a_out = self.bn3a_branch2a(self.res3a_branch2a(res2c_out))
        res3a_branch2a_relu = F.relu(res3a_branch2a_out)
        res3a_branch2b_out = self.bn3a_branch2b(self.res3a_branch2b(res3a_branch2a_relu))
        res3a_branch2b_relu = F.relu(res3a_branch2b_out)
        res3a_branch2c_out = self.bn3a_branch2c(self.res3a_branch2c(res3a_branch2b_relu))
        res3a_out = F.relu(res3a_branch1_out + res3a_branch2c_out)

        # Res3b
        res3b_branch2a_out = self.bn3b_branch2a(self.res3b_branch2a(res3a_out))
        res3b_branch2a_relu = F.relu(res3b_branch2a_out)
        res3b_branch2b_out = self.bn3b_branch2b(self.res3b_branch2b(res3b_branch2a_relu))
        res3b_branch2b_relu = F.relu(res3b_branch2b_out)
        res3b_branch2c_out = self.bn3b_branch2c(self.res3b_branch2c(res3b_branch2b_relu))
        res3b_out = F.relu(res3a_out + res3b_branch2c_out)

        # Global average pooling
        pool5_out = self.pool5(res3b_out)
        # output = pool5_out.view(pool5_out.size(0), -1)  # Flatten
        return torch.mean(res3b_out, dim=[2, 3], keepdim=False)
    def overwrite_init(self, device='cpu'):
        """
        Load pre-trained weights from a .npy file and assign to model.
        Skips layers containing 'fc11' or '/conv1/' in their name.
        """

        # Step 1: Load the .npy file (legacy TensorFlow format)
        np_var = {}

        def refine_np_var(input_data, output_dict, curr_tag=''):
            if isinstance(input_data, dict):
                for key, val in input_data.items():
                    if 'fc11' not in key:
                        refine_np_var(val, output_dict, curr_tag + '/' + key)
            else:
                if curr_tag not in output_dict:
                    output_dict[curr_tag] = input_data

        # Load pretrained weights
        pretrained_weights = np.load('crf_net_v2.npy', allow_pickle=True, encoding='latin1').item()
        refine_np_var(pretrained_weights, np_var)

        # Step 2: Map PyTorch parameters to the np_var keys
        def tf_name_2_torch_name(tf_key):
            # Remove scope prefix if any
            tf_key = tf_key.replace(f'{self.scope}/', '')
            # Replace BatchNorm names
            tf_key = tf_key.replace( 'offset','bias')
            tf_key = tf_key.replace('scale','weight')
            tf_key = tf_key.replace('mean','running_mean')
            tf_key = tf_key.replace( 'variance','running_var')
            tf_key = tf_key.replace( 'weights','weight')
            tf_key = tf_key.replace( '/','.')
            return tf_key[1:]

        # Build a mapping from transformed name -> parameter / buffer
        torch_vars = {}
        for name, param in self.named_parameters():
            torch_name = name
            
            torch_vars[torch_name] = param
        for name, buffer in self.named_buffers():
            torch_name = name
            
            torch_vars[torch_name] = buffer

        # Step 3: Load weights into model
        for key, value in np_var.items():
            if '/conv1/' in key:
                print(f"Skipping conv1 layer: {key}")
                continue
            torchkey=tf_name_2_torch_name(key)
            if torchkey in torch_vars:
                
                var = torch_vars[torchkey]
                value_tensor = torch.from_numpy(value).to(var.device if hasattr(var, 'device') else device)
                if 'weight' in torchkey and len(value_tensor.shape) == 4:
                    value_tensor = value_tensor.permute(3, 2, 0, 1)
                if var.shape != value_tensor.shape:
                    raise ValueError(f"Shape mismatch for {key} {torchkey}: {var.shape} vs {value_tensor.shape}")

                if isinstance(var, torch.nn.Parameter):
                    var.data.copy_(value_tensor)
                elif isinstance(var, torch.Tensor):
                    var.copy_(value_tensor)
                else:
                    raise TypeError(f"Unsupported type for {key} {torchkey}: {type(var)}")

                print(f"Loaded weight: {key}")
            else:
                print(f"Key not found in model: {key}  {torchkey}")
        # print(torch_vars)
        return
    

class AEInvcrfDecodeNet(nn.Module):

    def __init__(
            self,
            n_digit=2,
    ):
        # super().__init__()#'ae_invcrf_decode_net'
        super(AEInvcrfDecodeNet, self).__init__()

        self.n_digit = n_digit
        self.decode_spec = []
        self.s = 1024
        self.n_p = 12
        self.act = torch.tanh
        self.layers = nn.ModuleList()

        # Build decode_spec layers (same as dense layers)
        in_dim = 512
        for out_dim in self.decode_spec:
            self.layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.final_layer = nn.Linear(in_dim, self.n_p - 1)  # final output layer

        # Preload parsed constants from files
        _, G0, HINV = self.parse_invemor()
        self.register_buffer('G0', torch.tensor(G0, dtype=torch.float32))  # [s]
        self.register_buffer('HINV', torch.tensor(HINV, dtype=torch.float32))  # [s, 11]

    def parse_invemor(self):
        def _parse(lines, tag):
            for line_idx, line in enumerate(lines):
                if line == tag:
                    break
            s_idx = line_idx + 1
            r = []
            for idx in range(s_idx, s_idx + int(1024 / 4)):
                r += lines[idx].split()
            return np.float32(r)

        with open(os.path.join('invemor.txt'), 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

        b = _parse(lines, 'B =')
        g0 = _parse(lines, 'g0 =')
        hinv = np.stack([_parse(lines, 'hinv(%d)=' % (i + 1)) for i in range(11)], axis=-1)

        return b, g0, hinv

    def invcrf_pca_w_2_invcrf(self, invcrf_pca_w):
        """
        invcrf_pca_w: [b, 11]
        returns: [b, s]
        """
        batch_size = invcrf_pca_w.shape[0]

        # Expand G0 to [b, s, 1]
        G0 = self.G0.unsqueeze(0).unsqueeze(-1)  # [1, s, 1]
        G0 = G0.expand(batch_size, -1, -1)       # [b, s, 1]

        # Expand HINV to [b, s, 11]
        HINV = self.HINV.unsqueeze(0)           # [1, s, 11]
        HINV = HINV.expand(batch_size, -1, -1)   # [b, s, 11]

        # Reshape invcrf_pca_w to [b, 11, 1]
        invcrf_pca_w = invcrf_pca_w.unsqueeze(-1)  # [b, 11, 1]

        # Compute matmul: [b, s, 1]
        invcrf = G0 + torch.matmul(HINV, invcrf_pca_w)

        # Squeeze to [b, s]
        invcrf = invcrf.squeeze(-1)
        return invcrf

    def forward(self, feature):
        """
        Args:
            feature: Tensor of shape [b, n_digit]
        Returns:
            invcrf: Tensor of shape [b, s] where s = 1024
        """
        x = feature
        for layer in self.layers:
            x = layer(x)
            x = self.act(x)
        x = self.final_layer(x)  # [b, n_p - 1]

        invcrf = self.invcrf_pca_w_2_invcrf(x)
        return invcrf
    


class LinearizationNet(nn.Module):

    def __init__(self):
        # super().__init__('linearization_net')
        super(LinearizationNet, self).__init__()

        self.crf_feature_net = CrfFeatureNet()
        self.ae_invcrf_decode_net = AEInvcrfDecodeNet()

        self._register_child_modules()

    def _register_child_modules(self):
        for name, module in self.__dict__.items():
            if isinstance(module, nn.Module):
                self.add_module(name, module)

    @staticmethod
    def _resize_img(img, t):
        """
        Resize image to t x t, keeping aspect ratio.
        Input: img - Tensor of shape [B, C, H, W]
        Output: resized img - Tensor of shape [B, C, t, t]
        """
        B, C, H, W = img.shape
        ratio = H / W
        if ratio > 1.0:
            new_h = int(round(t * ratio))
            new_w = t
        else:
            new_h = t
            new_w = int(round(t / ratio))

        # Resize
        img = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # Center crop or pad to t x t
        transform = transforms.CenterCrop(t)
        result = torch.zeros((B, C, t, t), device=img.device)
        for i in range(B):
            result[i] = transform(img[i].unsqueeze(0)).squeeze(0)
        return result

    @staticmethod
    def _increase(rf):
        """
        Args:
            rf: Tensor of shape [B, 1024]

        Returns:
            new_rf: Tensor of shape [B, 1024], monotonic increasing
        """
        # g = rf[:, 1:] - rf[:, :-1]
        g = rf[:, 1:] - rf[:, :-1]  # [B, 1023]

        # min_g = tf.reduce_min(g, axis=-1, keep_dims=True)
        min_g = torch.min(g, dim=-1, keepdim=True)[0]  # [B, 1]

        # r = tf.nn.relu(-min_g)
        r = torch.relu(-min_g)  # [B, 1]

        # new_g = g + r.expand_as(g)
        # 注意：r 是 [B, 1]，g 是 [B, 1023]，需要扩展 r 到 [B, 1023]
        new_g = g + r.expand_as(g)  # [B, 1023]

        # new_g = new_g / tf.reduce_sum(new_g, axis=-1, keep_dims=True)
        sum_g = torch.sum(new_g, dim=-1, keepdim=True)  # [B, 1]
        new_g = new_g / (sum_g + 1e-8)  # 防止除以零

        # new_rf = tf.cumsum(new_g, axis=-1)
        new_rf = torch.cumsum(new_g, dim=-1)  # [B, 1023]

        # new_rf = tf.pad(new_rf, [[0, 0], [1, 0]], 'CONSTANT')
        new_rf = torch.cat([torch.zeros_like(new_rf[:, :1]), new_rf], dim=-1)  # [B, 1024]

        return new_rf

    def _get_edge(self, img):
        """
        Sobel edge detection using conv2d
        Input: img - [B, C, H, W]
        Output: edge_1 - [B, 6, H, W]
        """
        sobel_kernel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)

        sobel_kernel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)

        gray = torch.mean(img, dim=1, keepdim=True)
        gx = F.conv2d(gray, sobel_kernel_x, padding=1)
        gy = F.conv2d(gray, sobel_kernel_y, padding=1)

        edge_1 = torch.cat([gx, gy,
                            gx.abs(), gy.abs(),
                            gx.pow(2), gy.pow(2)], dim=1)  # [B, 6, H, W]

        return edge_1

    def _histogram_layer(self, img, max_bin):
        """
        Generate histogram features from input image
        Input: img - [B, C, H, W]
        Output: histogram_tensor - [B, C*(max_bin+1), H, W]
        """
        batch_size, channels, h, w = img.shape
        img = img.clamp(0.0, 1.0)
        bin_width = 1.0 / max_bin
        tmp_list = []
        for i in range(max_bin + 1):
            center = i * bin_width
            histo = torch.relu(1.0 - torch.abs(img - center) / bin_width)
            tmp_list.append(histo)
        histogram_tensor = torch.cat(tmp_list, dim=1)
        return histogram_tensor

    def forward(self, img, is_training=True):
        """
        Main forward pass
        Input: img - [B, 3, H, W]
        Output: invcrf - [B, 1024]
        """

        # Edge branch
        edge_1 = self._get_edge(img)

        # Histogram branches
        hist_4 = self._histogram_layer(img, 4)
        hist_8 = self._histogram_layer(img, 8)
        hist_16 = self._histogram_layer(img, 16)

        # Concat all inputs
        concat_input = torch.cat([img, edge_1, hist_4, hist_8, hist_16], dim=1)

        # Feature extraction
        feature = self.crf_feature_net(concat_input, is_training=is_training)

        # Decode to inverse CRF
        invcrf = self.ae_invcrf_decode_net(feature)

        # Ensure monotonic increase
        invcrf = self._increase(invcrf)

        return invcrf

    def get_output(self, img, is_training=True):
        return self(img, is_training)
    

if __name__ == '__main__':
    model=LinearizationNet()
    # 加载前打印
    print(model.crf_feature_net.bn2a_branch1.weight.data)
    model.crf_feature_net.overwrite_init()
    print(model.crf_feature_net.bn2a_branch1.weight.data)