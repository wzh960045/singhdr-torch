import torch
import torch.nn as nn
import torch.nn.functional as F

class Dequantization_net(nn.Module):
    def __init__(self):
        super(Dequantization_net, self).__init__()
        
        # Define layers in the initialization to ensure they are properly registered and updated during training
        self.conv_init_1 = nn.Conv2d(3, 16, 7, padding=3)
        self.conv_s1_1 = nn.Conv2d(16, 16, 7, padding=3)
        self.down_conv_s2_1 = nn.Conv2d(16, 32, 5, padding=2)
        self.down_conv_s2_2 = nn.Conv2d(32, 32, 5, padding=2)
        self.down_conv_s3_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.down_conv_s3_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.down_conv_s4_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.down_conv_s4_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.down_conv_x_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.down_conv_x_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.up_conv_x_1 = nn.Conv2d(256 + 128, 128, 3, padding=1)
        self.up_conv_x_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.up_conv_s4_1 = nn.Conv2d(128 + 64, 64, 3, padding=1)
        self.up_conv_s4_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.up_conv_s3_1 = nn.Conv2d(64 + 32, 32, 3, padding=1)
        self.up_conv_s3_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.up_conv_s2_1 = nn.Conv2d(32 + 16, 16, 3, padding=1)
        self.up_conv_s2_2 = nn.Conv2d(16, 16, 3, padding=1)
        self.final_conv = nn.Conv2d(16, 3, 3, padding=1)

    def inference(self, input_images):
        return self._build_model(input_images)
    
    def forward(self, input_images):
        return self._build_model(input_images)
    
    def loss(self, predictions, targets):
        return F.mse_loss(predictions, targets)
    
    def down(self, x, conv1, conv2):
        x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(conv1(x), 0.1)
        x = F.leaky_relu(conv2(x), 0.1)
        return x
    
    def up(self, x, conv1, conv2, skpCn):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.leaky_relu(conv1(torch.cat([x, skpCn], dim=1)), 0.1)
        x = F.leaky_relu(conv2(x), 0.1)
        return x
    
    def _build_model(self, input_images):
        print(input_images.shape)
        
        x = F.leaky_relu(self.conv_init_1(input_images), 0.1)
        s1 = F.leaky_relu(self.conv_s1_1(x), 0.1)
        s2 = self.down(s1, self.down_conv_s2_1, self.down_conv_s2_2)
        s3 = self.down(s2, self.down_conv_s3_1, self.down_conv_s3_2)
        s4 = self.down(s3, self.down_conv_s4_1, self.down_conv_s4_2)
        x = self.down(s4, self.down_conv_x_1, self.down_conv_x_2)
        x = self.up(x, self.up_conv_x_1, self.up_conv_x_2, s4)
        x = self.up(x, self.up_conv_s4_1, self.up_conv_s4_2, s3)
        x = self.up(x, self.up_conv_s3_1, self.up_conv_s3_2, s2)
        x = self.up(x, self.up_conv_s2_1, self.up_conv_s2_2, s1)
        x = torch.tanh(self.final_conv(x))
        output = input_images + x
        return output
    