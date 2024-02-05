from tinygrad import Tensor, nn

t = Tensor.rand((1, 32, 128, 128, 128))
conv = nn.Conv2d(32, 32, kernel_size=(3,3,3), padding=(1,1,1,1,1,1), bias=False)

# print("Realize conv")
# conv_out = conv(t).realize()

# mit wenig ram -> fail
# mit viel ram -> success
# mit viel ram & seq -> ?
print("Realize multiple")

a = conv(t)
b = conv(a)
c = conv(b)
print(c.realize().shape)

# print("Realize seq")
# t.sequential([conv, conv, conv]).realize()
