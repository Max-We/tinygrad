from tinygrad import Tensor, nn

t = Tensor.rand(128, 128, 256)
conv = nn.Conv2d(256, 128, (3, 3), padding=1)
print(conv(t).realize())