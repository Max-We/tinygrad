from tinygrad import Tensor, nn

t = Tensor.rand(256, 128, 128)
conv = nn.Conv2d(256, 128, (3, 3), padding=1)
print(conv.realize())