import torch
import torch.nn as nn

# Assuming input tensor shape is (batch_size, channels, depth, height, width)
input_tensor = torch.randn((1, 32, 128, 128, 128))

# Define 3D convolution layer
conv = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), padding=(1, 1, 1))

# Apply convolution
conv_out = conv(input_tensor)

# Print the result
print(conv_out.shape)
