from tinygrad import Tensor, nn

t = Tensor.rand(2, 256, 300, 300)
conv = nn.Conv2d(256, 128, (3, 3), padding=1)

print("Realize")
wow = conv(t).realize()
print("Done", wow.shape)