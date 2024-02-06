from tinygrad import Tensor, nn, TinyJit, Device

Device.DEFAULT="CPU"
channels = 32

@TinyJit
def do_ops():
  with Tensor.train:
    t = Tensor.rand((1, channels, 128, 128, 128))
    conv = nn.Conv2d(channels, channels, kernel_size=(3,3,3), padding=(1,1,1,1,1,1), bias=False)
    norm = nn.InstanceNorm(channels)
    relu = Tensor.relu

    seq = [conv, norm, relu]

    print("Realize conv 1")
    conv_out = conv(t).realize()

    print(conv_out.shape)

do_ops()

print("Done")