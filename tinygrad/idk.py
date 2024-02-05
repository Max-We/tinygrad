from tinygrad import Tensor, nn

t = Tensor.rand((1, 32, 128, 128, 128))
conv = nn.Conv2d(32, 32, kernel_size=(3,3,3), padding=(1,1,1,1,1,1), bias=False)
norm = nn.InstanceNorm(32)
relu = Tensor.relu

seq = [conv, norm, relu]

# print("Realize conv")
# conv_out = conv(t).realize()

# mit wenig ram -> fail
# mit viel ram -> success
# mit viel ram & seq -> ?
print("Realize multiple")

a = t.sequential(seq)
b = t.sequential(a)
c = t.sequential(b)
print(c.realize().shape)

# print("Realize seq")
# t.sequential([conv, conv, conv]).realize()
