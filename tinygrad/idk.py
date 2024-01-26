from tinygrad import Tensor, nn

t = Tensor.rand((1, 1, 128, 128, 128))
conv = nn.Conv2d(1, 32, kernel_size=(3,3,3), stride=1, padding=(1,1,1,1,1,1), bias=False)
norm = nn.InstanceNorm(32)
seq = [conv, norm, Tensor.relu]

print("Realize conv")
conv_out = conv(t).realize()

print("Realize norm")
norm_out = norm(conv_out)

print("Realize relu")
norm_out.relu().realize()

print("All done")

print("Realize sequential")
print(t.sequential(seq).realize())
