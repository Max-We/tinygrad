import torch
import torch.nn as nn

torch.backends.cudnn.deterministic = True

# Assuming input tensor shape is (batch_size, channels, depth, height, width)
channels = 320
input_tensor = torch.randn((1, channels, 128, 128, 128))

# Define 3D convolution layer
conv = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

# Apply convolution
comp = torch.compile(conv, backend="eager")
conv_out = comp(input_tensor)

# Print the result
print(conv_out.shape)
