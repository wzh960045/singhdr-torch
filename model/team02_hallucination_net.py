"""
 " License:
 " -----------------------------------------------------------------------------
 " Copyright (c) 2017, Gabriel Eilertsen.
 " All rights reserved.
 "
 " Redistribution and use in source and binary forms, with or without
 " modification, are permitted provided that the following conditions are met:
 "
 " 1. Redistributions of source code must retain the above copyright notice,
 "    this list of conditions and the following disclaimer.
 "
 " 2. Redistributions in binary form must reproduce the above copyright notice,
 "    this list of conditions and the following disclaimer in the documentation
 "    and/or other materials provided with the distribution.
 "
 " 3. Neither the name of the copyright holder nor the names of its contributors
 "    may be used to endorse or promote products derived from this software
 "    without specific prior written permission.
 "
 " THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 " AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 " IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 " ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 " LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 " CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 " SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 " INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 " CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 " ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 " POSSIBILITY OF SUCH DAMAGE.
 " -----------------------------------------------------------------------------
 "
 " Description: TensorFlow autoencoder CNN for HDR image reconstruction.
 " Author: Gabriel Eilertsen, gabriel.eilertsen@liu.se
 " Date: Aug 2017
"""

# import tensorflow as tf
# import tensorlayer as tl
# import numpy as np


# # The HDR reconstruction autoencoder fully convolutional neural network
# def model(x, batch_size=1, is_training=False):
#     # Encoder network (VGG16, until pool5)
#     x_in = tf.scalar_mul(255.0, x)
#     net_in = tl.layers.InputLayer(x_in, name='input_layer')
#     conv_layers, skip_layers = encoder(net_in)

#     # Fully convolutional layers on top of VGG16 conv layers
#     network = tl.layers.Conv2dLayer(conv_layers,
#                                     act=tf.identity,
#                                     shape=[3, 3, 512, 512],
#                                     strides=[1, 1, 1, 1],
#                                     padding='SAME',
#                                     name='encoder/h6/conv')
#     #network = tf.layers.batch_normalization(network, training=is_training, name='encoder/h6/batch_norm')
#     network = tl.layers.BatchNormLayer(network, is_train=is_training, name='encoder/h6/batch_norm')
#     network.outputs = tf.nn.relu(network.outputs, name='encoder/h6/relu')

#     # Decoder network
#     network = decoder(network, skip_layers, batch_size, is_training)

#     """if is_training:
#         return network, conv_layers"""

#     return network, conv_layers


# # Final prediction of the model, including blending with input
# def get_finaltf(network, x_in):
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


# # Convolutional layers of the VGG16 model used as encoder network
# def encoder(input_layer):
#     VGG_MEAN = [103.939, 116.779, 123.68]

#     # Convert RGB to BGR
#     red, green, blue = tf.split(input_layer.outputs, 3, 3)
#     bgr = tf.concat([blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]], axis=3)

#     network = tl.layers.InputLayer(bgr, name='encoder/input_layer_bgr')

#     # Convolutional layers size 1
#     network = conv_layer(network, [3, 64], 'encoder/h1/conv_1')
#     beforepool1 = conv_layer(network, [64, 64], 'encoder/h1/conv_2')
#     network = pool_layer(beforepool1, 'encoder/h1/pool')

#     # Convolutional layers size 2
#     network = conv_layer(network, [64, 128], 'encoder/h2/conv_1')
#     beforepool2 = conv_layer(network, [128, 128], 'encoder/h2/conv_2')
#     network = pool_layer(beforepool2, 'encoder/h2/pool')

#     # Convolutional layers size 3
#     network = conv_layer(network, [128, 256], 'encoder/h3/conv_1')
#     network = conv_layer(network, [256, 256], 'encoder/h3/conv_2')
#     beforepool3 = conv_layer(network, [256, 256], 'encoder/h3/conv_3')
#     network = pool_layer(beforepool3, 'encoder/h3/pool')

#     # Convolutional layers size 4
#     network = conv_layer(network, [256, 512], 'encoder/h4/conv_1')
#     network = conv_layer(network, [512, 512], 'encoder/h4/conv_2')
#     beforepool4 = conv_layer(network, [512, 512], 'encoder/h4/conv_3')
#     network = pool_layer(beforepool4, 'encoder/h4/pool')

#     # Convolutional layers size 5
#     network = conv_layer(network, [512, 512], 'encoder/h5/conv_1')
#     network = conv_layer(network, [512, 512], 'encoder/h5/conv_2')
#     beforepool5 = conv_layer(network, [512, 512], 'encoder/h5/conv_3')
#     network = pool_layer(beforepool5, 'encoder/h5/pool')

#     return network, (input_layer, beforepool1, beforepool2, beforepool3, beforepool4, beforepool5)


# # Decoder network
# def decoder(input_layer, skip_layers, batch_size=1, is_training=False):
#     sb, sx, sy, sf = input_layer.outputs.get_shape().as_list()
#     alpha = 0.0

#     # Upsampling 1
#     network = deconv_layer(input_layer, (batch_size, sx, sy, sf, sf), 'decoder/h1/decon2d', alpha, is_training)

#     # Upsampling 2
#     network = skip_connection_layer(network, skip_layers[5], 'decoder/h2/fuse_skip_connection', is_training)
#     network = deconv_layer(network, (batch_size, sx, sy, sf, sf), 'decoder/h2/decon2d', alpha, is_training)

#     # Upsampling 3
#     network = skip_connection_layer(network, skip_layers[4], 'decoder/h3/fuse_skip_connection', is_training)
#     network = deconv_layer(network, (batch_size, sx, sy, sf, sf / 2), 'decoder/h3/decon2d', alpha, is_training)

#     # Upsampling 4
#     network = skip_connection_layer(network, skip_layers[3], 'decoder/h4/fuse_skip_connection', is_training)
#     network = deconv_layer(network, (batch_size, sx, sy, sf / 2, sf / 4), 'decoder/h4/decon2d', alpha,
#                            is_training)

#     # Upsampling 5
#     network = skip_connection_layer(network, skip_layers[2], 'decoder/h5/fuse_skip_connection', is_training)
#     network = deconv_layer(network, (batch_size, sx, sy, sf / 4, sf / 8), 'decoder/h5/decon2d', alpha,
#                            is_training)

#     # Skip-connection at full size
#     network = skip_connection_layer(network, skip_layers[1], 'decoder/h6/fuse_skip_connection', is_training)

#     # Final convolution
#     network = tl.layers.Conv2dLayer(network,
#                                     act=tf.identity,
#                                     shape=[1, 1, int(sf / 8), 3],
#                                     strides=[1, 1, 1, 1],
#                                     padding='SAME',
#                                     W_init=tf.contrib.layers.xavier_initializer(uniform=False),
#                                     b_init=tf.constant_initializer(value=0.0),
#                                     name='decoder/h7/conv2d')

#     # Final skip-connection
#     network = tl.layers.BatchNormLayer(network, is_train=is_training, name='decoder/h7/batch_norm')
#     network.outputs = tf.maximum(alpha * network.outputs, network.outputs, name='decoder/h7/leaky_relu')
#     network = skip_connection_layer(network, skip_layers[0], 'decoder/h7/fuse_skip_connection')

#     return network


# # Load weights for VGG16 encoder convolutional layers
# # Weights are from a .npy file generated with the caffe-tensorflow tool
# def load_vgg_weights(network, weight_file, session):
#     params = []

#     if weight_file.lower().endswith('.npy'):
#         npy = np.load(weight_file, encoding='latin1')
#         for key, val in sorted(npy.item().items()):
#             if (key[:4] == "conv"):
#                 print("  Loading %s" % (key))
#                 print("  weights with size %s " % str(val['weights'].shape))
#                 print("  and biases with size %s " % str(val['biases'].shape))
#                 params.append(val['weights'])
#                 params.append(val['biases'])
#     else:
#         print('No weights in suitable .npy format found for path ', weight_file)

#     print('Assigning loaded weights..')
#     tl.files.assign_params(session, params, network)

#     return network


# # === Layers ==================================================================

# # Convolutional layer
# def conv_layer(input_layer, sz, str):
#     network = tl.layers.Conv2dLayer(input_layer,
#                                     act=tf.nn.relu,
#                                     shape=[3, 3, sz[0], sz[1]],
#                                     strides=[1, 1, 1, 1],
#                                     padding='SAME',
#                                     name=str)

#     return network


# # Max-pooling layer
# def pool_layer(input_layer, str):
#     network = tl.layers.PoolLayer(input_layer,
#                                   ksize=[1, 2, 2, 1],
#                                   strides=[1, 2, 2, 1],
#                                   padding='SAME',
#                                   pool=tf.nn.max_pool,
#                                   name=str)

#     return network


# # Concatenating fusion of skip-connections
# def skip_connection_layer(input_layer, skip_layer, str, is_training=False):
#     _, sx, sy, sf = input_layer.outputs.get_shape().as_list()
#     _, sx_, sy_, sf_ = skip_layer.outputs.get_shape().as_list()

#     #assert (sx_, sy_, sf_) == (sx, sy, sf)

#     # skip-connection domain transformation, from LDR encoder to log HDR decoder
#     # skip_layer.outputs = tf.log(tf.pow(tf.scalar_mul(1.0 / 255, skip_layer.outputs), 2.0) + 1.0 / 255.0)
#     skip_layer.outputs = tf.scalar_mul(1.0 / 255, skip_layer.outputs)

#     # specify weights for fusion of concatenation, so that it performs an element-wise addition
#     weights = np.zeros((1, 1, sf + sf_, sf))
#     for i in range(sf):
#         weights[0, 0, i, i] = 1
#         weights[:, :, i + sf_, i] = 1
#     add_init = tf.constant_initializer(value=weights, dtype=tf.float32)

#     # concatenate layers
#     network = tl.layers.ConcatLayer([input_layer, skip_layer], concat_dim=3, name='%s/skip_connection' % str)

#     # fuse concatenated layers using the specified weights for initialization
#     network = tl.layers.Conv2dLayer(network,
#                                     act=tf.identity,
#                                     shape=[1, 1, sf + sf_, sf],
#                                     strides=[1, 1, 1, 1],
#                                     padding='SAME',
#                                     W_init=add_init,
#                                     b_init=tf.constant_initializer(value=0.0),
#                                     name=str)

#     return network


# # Deconvolution layer
# def deconv_layer(input_layer, sz, str, alpha, is_training=False):
#     scale = 2

#     filter_size = (2 * scale - scale % 2)
#     num_in_channels = int(sz[3])
#     num_out_channels = int(sz[4])

#     network = tl.layers.UpSampling2dLayer(input_layer, (scale, scale), True, 1, False, '%s/NN_dc' % str)
#     network = tl.layers.PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT", name='inpad')
#     # network = conv_layer(network, [num_in_channels, num_out_channels], str)
#     network = tl.layers.Conv2dLayer(network,
#                                     act=tf.nn.relu,
#                                     shape=[3, 3, num_in_channels, num_out_channels],
#                                     strides=[1, 1, 1, 1],
#                                     padding='VALID',
#                                     name=str)

#     network = tl.layers.BatchNormLayer(network, is_train=is_training, name='%s/batch_norm_dc' % str)
#     network.outputs = tf.maximum(alpha * network.outputs, network.outputs, name='%s/leaky_relu_dc' % str)


#     return network



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Convolutional layer
class ConvLayer(nn.Module):
    def __init__(self, sz, str):
        """
        Args:
            sz (tuple or list): [in_channels, out_channels]
            str (str): layer name (not used in PyTorch, but kept for reference)
        """
        super(ConvLayer, self).__init__()
        self.name = str  # store name if needed
        self.conv = nn.Conv2d(
            in_channels=sz[0],
            out_channels=sz[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        return x


# Max-pooling layer
class PoolLayer(nn.Module):
    def __init__(self, str):
        super(PoolLayer, self).__init__()
        self.name = str
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.pool(x)
        return x
    



# Skip Connection Fusion Layer
class SkipConnectionLayer(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels=None, name=None):
        """
        Args:
            in_channels (int): number of input feature maps
            skip_channels (int): number of skip feature maps
            out_channels (int): output feature maps. Default: same as in_channels
            name (str): optional name for the layer
        """
        super(SkipConnectionLayer, self).__init__()
        self.name = name or "skip_connection"

        if out_channels is None:
            out_channels = in_channels

        # Initialize weights to perform addition-like fusion
        weight_init = torch.zeros(1, 1, in_channels + skip_channels, out_channels)
        for i in range(out_channels):
            weight_init[0, 0, i, i] = 1.0
            if i < skip_channels:
                weight_init[0, 0, i + in_channels, i] = 1.0

        self.conv_fuse = nn.Conv2d(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        # Set custom weights
        self.conv_fuse.weight.data = weight_init.permute(3, 2, 0, 1)  # [out, in, H, W]

    def forward(self, x, skip):
        """
        x: Input tensor (from decoder path)
        skip: Skip connection tensor (from encoder path)
        """
        # Normalize skip connection (optional preprocessing)
        skip = skip / 255.0  # equivalent to scalar_mul

        # Concatenate along channel dimension
        combined = torch.cat([x, skip], dim=1)

        # Fuse using learned convolution
        fused = self.conv_fuse(combined)

        return fused


# Deconvolution Layer (Upsample + Conv + BatchNorm + LeakyReLU)
class DeconvLayer(nn.Module):
    def __init__(self, sz, alpha=0.2, name=None):
        """
        Args:
            sz (tuple or list): [in_channels, out_channels]
            alpha (float): slope for leaky ReLU
            name (str): optional name for the layer
        """
        super(DeconvLayer, self).__init__()
        self.name = name or "deconv_layer"

        in_channels, out_channels = sz[0], sz[1]

        # Upsample and pad manually
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.pad = nn.ReflectionPad2d(1)  # Reflect pad [top, bottom, left, right]

        # Conv layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=0,  # Valid padding after manual pad
            bias=False
        )

        # BatchNorm
        self.bn = nn.BatchNorm2d(out_channels)

        # Leaky ReLU
        self.alpha = alpha

    def forward(self, x):
        x = self.upsample(x)
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)

        # LeakyReLU manually
        x = torch.where(x > 0, x, x * self.alpha)

        return x





class VGGEcoder(nn.Module):
    def __init__(self):
        super(VGGEcoder, self).__init__()

        # VGG 均值 (BGR)
        self.register_buffer('VGG_MEAN', torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1))

        # 各阶段的 skip connections 会返回
        self.conv1_1 = ConvLayer(sz=[3, 64], str='encoder/h1/conv_1')
        self.conv1_2 = ConvLayer(sz=[64, 64], str='encoder/h1/conv_2')
        self.pool1 = PoolLayer(str='encoder/h1/pool')

        self.conv2_1 = ConvLayer(sz=[64, 128], str='encoder/h2/conv_1')
        self.conv2_2 = ConvLayer(sz=[128, 128], str='encoder/h2/conv_2')
        self.pool2 = PoolLayer(str='encoder/h2/pool')

        self.conv3_1 = ConvLayer(sz=[128, 256], str='encoder/h3/conv_1')
        self.conv3_2 = ConvLayer(sz=[256, 256], str='encoder/h3/conv_2')
        self.conv3_3 = ConvLayer(sz=[256, 256], str='encoder/h3/conv_3')
        self.pool3 = PoolLayer(str='encoder/h3/pool')

        self.conv4_1 = ConvLayer(sz=[256, 512], str='encoder/h4/conv_1')
        self.conv4_2 = ConvLayer(sz=[512, 512], str='encoder/h4/conv_2')
        self.conv4_3 = ConvLayer(sz=[512, 512], str='encoder/h4/conv_3')
        self.pool4 = PoolLayer(str='encoder/h4/pool')

        self.conv5_1 = ConvLayer(sz=[512, 512], str='encoder/h5/conv_1')
        self.conv5_2 = ConvLayer(sz=[512, 512], str='encoder/h5/conv_2')
        self.conv5_3 = ConvLayer(sz=[512, 512], str='encoder/h5/conv_3')
        self.pool5 = PoolLayer(str='encoder/h5/pool')

    def forward(self, x):
        """
        Input: x - tensor of shape [N, 3, H, W] in RGB format
        Output:
            features: final output feature map after last pooling
            skips: tuple of skip connection feature maps
        """

        # Convert RGB to BGR and subtract VGG mean
        red, green, blue = x.split(1, dim=1)
        bgr = torch.cat([blue, green, red], dim=1)  # Swap channels to BGR
        bgr = bgr - self.VGG_MEAN  # Subtract VGG mean

        # Block 1
        h1_conv1 = self.conv1_1(bgr)
        h1_conv2 = self.conv1_2(h1_conv1)
        pool1 = self.pool1(h1_conv2)

        # Block 2
        h2_conv1 = self.conv2_1(pool1)
        h2_conv2 = self.conv2_2(h2_conv1)
        pool2 = self.pool2(h2_conv2)

        # Block 3
        h3_conv1 = self.conv3_1(pool2)
        h3_conv2 = self.conv3_2(h3_conv1)
        h3_conv3 = self.conv3_3(h3_conv2)
        pool3 = self.pool3(h3_conv3)

        # Block 4
        h4_conv1 = self.conv4_1(pool3)
        h4_conv2 = self.conv4_2(h4_conv1)
        h4_conv3 = self.conv4_3(h4_conv2)
        pool4 = self.pool4(h4_conv3)

        # Block 5
        h5_conv1 = self.conv5_1(pool4)
        h5_conv2 = self.conv5_2(h5_conv1)
        h5_conv3 = self.conv5_3(h5_conv2)
        pool5 = self.pool5(h5_conv3)

        # 返回最后一层输出和所有 skip connection 的中间输出
        return pool5, (x, h1_conv2, h2_conv2, h3_conv3, h4_conv3, h5_conv3)
    



class VGDecoder(nn.Module):
    def __init__(self, base_channels=512, alpha=0.0):
        """
        Args:
            base_channels (int): number of channels at the bottleneck (e.g., 512 for VGG)
            alpha (float): negative slope for leaky ReLU
        """
        super(VGDecoder, self).__init__()
        self.alpha = alpha

        # 初始化 decoder 层
        self.deconv1 = DeconvLayer(sz=(base_channels, base_channels), name='decoder/h1/decon2d')
        self.skip1 = SkipConnectionLayer(in_channels=base_channels, skip_channels=base_channels, out_channels=base_channels)

        self.deconv2 = DeconvLayer(sz=(base_channels, base_channels), name='decoder/h2/decon2d')
        self.skip2 = SkipConnectionLayer(in_channels=base_channels, skip_channels=base_channels, out_channels=base_channels)

        self.deconv3 = DeconvLayer(sz=(base_channels, base_channels // 2), name='decoder/h3/decon2d')
        self.skip3 = SkipConnectionLayer(in_channels=base_channels // 2, skip_channels=base_channels // 2,
                                         out_channels=base_channels // 2)

        self.deconv4 = DeconvLayer(sz=(base_channels // 2, base_channels // 4), name='decoder/h4/decon2d')
        self.skip4 = SkipConnectionLayer(in_channels=base_channels // 4, skip_channels=base_channels // 4,
                                         out_channels=base_channels // 4)

        self.deconv5 = DeconvLayer(sz=(base_channels // 4, base_channels // 8), name='decoder/h5/decon2d')
        self.skip5 = SkipConnectionLayer(in_channels=base_channels // 8, skip_channels=base_channels // 8,
                                         out_channels=base_channels // 8)

        self.skip6 = SkipConnectionLayer(in_channels=3, skip_channels=3,
                                         out_channels=3)

        # Final conv layer to output RGB image
        self.final_conv = nn.Conv2d(
            in_channels=base_channels // 8,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        # Xavier 初始化卷积核
        nn.init.xavier_normal_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

        # BatchNorm and LeakyReLU
        self.final_bn = nn.BatchNorm2d(3)

    def forward(self, x, skip_layers):
        """
        Args:
            x: input tensor from encoder (bottleneck feature map)
            skip_layers: list of skip connection tensors from encoder
        Returns:
            output tensor (reconstructed image)
        """

        # Upsample 1
        x = self.deconv1(x)  # [B, 512, H/32, W/32] -> [B, 512, H/16, W/16]

        # Upsample 2
        x = self.skip1(x, skip_layers[5])  # fuse with encoder skip
        x = self.deconv2(x)

        # Upsample 3
        x = self.skip2(x, skip_layers[4])
        x = self.deconv3(x)

        # Upsample 4
        x = self.skip3(x, skip_layers[3])
        x = self.deconv4(x)

        # Upsample 5
        x = self.skip4(x, skip_layers[2])
        x = self.deconv5(x)

        # Skip connection at full resolution
        x = self.skip5(x, skip_layers[1])
        x = self.final_conv(x)
        

        # Final convolution
        x = self.final_bn(x)
        x = torch.where(x > 0, x, x * self.alpha)  # LeakyReLU
        x = self.skip6(x, skip_layers[0])

        return x

import torch
import torch.nn as nn




class HDRAutoencoder(nn.Module):
    def __init__(self, base_channels=512, alpha=0.0):
        """
        Args:
            base_channels (int): number of channels at the bottleneck (e.g., 512 for VGG)
            alpha (float): negative slope for leaky ReLU
        """
        super(HDRAutoencoder, self).__init__()

        # Encoder
        self.encoder = VGGEcoder()

        # Additional conv layer on top of encoder output (VGG head extension)
        self.top_conv = nn.Conv2d(
            in_channels=base_channels,
            out_channels=base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.top_bn = nn.BatchNorm2d(base_channels)

        # Decoder
        self.decoder = VGDecoder(base_channels=base_channels, alpha=alpha)

        # 初始化 top_conv 权重
        nn.init.kaiming_normal_(self.top_conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, is_training=True):
        """
        Args:
            x: input tensor (normalized to [0, 1] range)
            is_training: flag for batch norm and dropout behavior
        Returns:
            reconstructed image tensor
        """

        # Convert input from [0, 1] to [0, 255]
        x_in = x * 255.0

        # Encoder
        conv_layers, skip_layers = self.encoder(x_in)

        # Top convolutional layer (extension of VGG)
        features = self.top_conv(conv_layers)
        features = self.top_bn(features)
        features = F.relu(features, inplace=True)

        # Decoder
        output = self.decoder(features, skip_layers)

        return output, conv_layers  # 返回重建图像和中间特征图用于 loss 计算等
    


def get_final(network_output, x_in):
    """
    Args:
        network_output (torch.Tensor): [B, 3, H, W] 输出的预测 log HDR 图像（log 域）
        x_in (torch.Tensor): [B, 3, H, W] 输入的 LDR 图像（范围 [0, 1] 或 [0, 255]）

    Returns:
        y_final (torch.Tensor): [B, 3, H, W] 最终融合后的 HDR 预测图像
    """
    # 获取输入尺寸
    B, C, H, W = x_in.shape

    # Highlight mask threshold
    thr = 0.05

    # 计算每个像素的最大通道值作为亮度参考
    x_in_max, _ = x_in.max(dim=1, keepdim=True)  # [B, 1, H, W]

    # 构造 alpha mask
    alpha = torch.clamp((x_in_max - 1.0 + thr) / thr, min=0.0, max=1.0)  # [B, 1, H, W]

    # 扩展 alpha 到 3 通道
    alpha = alpha.expand(-1, 3, -1, -1)  # [B, 3, H, W]

    # Linearize input and convert network output from log domain to linear HDR
    x_lin = x_in.pow(2)                         # 模拟相机响应曲线（gamma矫正）
    y_predict = torch.exp(network_output) - 1.0 / 255.0  # 从 log 域还原 HDR

    # Alpha blending
    y_final = (1.0 - alpha) * x_lin + alpha * y_predict

    return y_final


from collections import OrderedDict


def load_vgg_weights(model: torch.nn.Module, weight_file: str):
    """
    Load pre-trained VGG16 weights from a .npy file (generated by caffe-tensorflow)
    and assign them to the PyTorch model.

    Args:
        model (torch.nn.Module): The encoder model (e.g., VGGEcoder)
        weight_file (str): Path to the .npy weight file

    Returns:
        model (torch.nn.Module): Model with loaded weights
    """

    if not weight_file.lower().endswith('.npy'):
        raise ValueError(f"Only .npy files are supported. Got: {weight_file}")

    # Load weights from .npy file
    try:
        vgg_weights = np.load(weight_file, allow_pickle=True, encoding='latin1').item()
    except Exception as e:
        raise IOError(f"Failed to load weights from {weight_file}: {e}")

    # Get all ConvLayer modules in the model (in order)
    conv_layers = [module for module in model.modules() if isinstance(module, ConvLayer)]
    
    print("=> Loading VGG16 weights into PyTorch model:")

    idx = 0  # index over conv layers
    for key in sorted(vgg_weights.keys()):
        if key.startswith('conv') and 'sub' not in key:  # skip sub-blocks if any
            weights = vgg_weights[key]['weights']
            biases = vgg_weights[key]['biases']

            print(f"  Loading {key} -> ConvLayer {idx}")
            print(f"    Weights shape: {weights.shape} -> Expected: {conv_layers[idx].conv.weight.shape}")
            print(f"    Biases shape: {biases.shape} -> Expected: {conv_layers[idx].conv.bias.shape}")

            # Convert weights from HWCN to NCHW format
            # TensorFlow: [H, W, C_in, C_out] => PyTorch: [C_out, C_in, H, W]
            weights_pt = torch.from_numpy(weights.transpose((3, 2, 0, 1)))
            biases_pt = torch.from_numpy(biases)

            # Assign weights and bias
            with torch.no_grad():
                conv_layers[idx].conv.weight.copy_(weights_pt)
                conv_layers[idx].conv.bias.copy_(biases_pt)

            idx += 1

            if idx > len(conv_layers):
                print(f"  Loading {key}")
                print(len(conv_layers),idx)
                print("Warning: More VGG layers found than ConvLayer modules in model.")
                

    if idx < len(conv_layers):
        print(f"Warning: Only {idx}/{len(conv_layers)} ConvLayers were initialized.")

    return model

if __name__ == '__main__':
    model=HDRAutoencoder()
    print(model.encoder.conv1_1.conv.weight.data)
    load_vgg_weights(model.encoder,'vgg16_places365_weights.npy')
    print(model.encoder.conv1_1.conv.weight.data)
    