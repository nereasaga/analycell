import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # función helper para crear bloques conv-relu
        def make_conv_block(input_channels, output_channels):
            block = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            return block
        
        # encoder layers
        self.enc1 = make_conv_block(1, 16)
        self.enc2 = make_conv_block(16, 32)
        
        # pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # bottleneck
        self.bottleneck = make_conv_block(32, 64)
        
        # decoder layers
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = make_conv_block(64, 32)  # 64 porque concatenamos
        
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = make_conv_block(32, 16)  # 32 porque concatenamos
        
        # output layer
        self.out = nn.Conv2d(16, 1, kernel_size=1)

    def crop_tensor_to_fit(self, tensor_to_crop, target_tensor):
        # obtener dimensiones
        batch_size, channels, height, width = tensor_to_crop.shape
        target_batch, target_channels, target_height, target_width = target_tensor.shape
        
        # calcular diferencias
        height_diff = height - target_height
        width_diff = width - target_width
        
        # hacer crop si es necesario
        if height_diff != 0 or width_diff != 0:
            start_h = height_diff // 2
            end_h = height - (height_diff - height_diff // 2)
            start_w = width_diff // 2
            end_w = width - (width_diff - width_diff // 2)
            
            cropped_tensor = tensor_to_crop[:, :, start_h:end_h, start_w:end_w]
            return cropped_tensor
        else:
            return tensor_to_crop

    def forward(self, x):
        # encoder path
        conv1 = self.enc1(x)
        pool1 = self.pool(conv1)
        
        conv2 = self.enc2(pool1)
        pool2 = self.pool(conv2)
        
        # bottleneck
        bottleneck = self.bottleneck(pool2)

        # decoder path
        up2 = self.up2(bottleneck)
        conv2_cropped = self.crop_tensor_to_fit(conv2, up2)
        concat2 = torch.cat([up2, conv2_cropped], dim=1)
        conv3 = self.dec2(concat2)

        up1 = self.up1(conv3)
        conv1_cropped = self.crop_tensor_to_fit(conv1, up1)
        concat1 = torch.cat([up1, conv1_cropped], dim=1)
        conv4 = self.dec1(concat1)

        # output
        output = self.out(conv4)
        return output