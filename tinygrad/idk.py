from tinygrad import Tensor, nn

t = Tensor.rand((1, 1, 128, 128, 128))
conv = nn.Conv2d(1, 32, kernel_size=(3,3,3), stride=1, padding=(1,1,1,1,1,1), bias=False)

print("Realize")
wow = conv(t).realize()
print("Done", wow.shape)