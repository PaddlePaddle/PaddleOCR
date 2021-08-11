from paddle import nn

class MTB(nn.Layer):
    def __init__(self, cnn_num, in_channels):
        super(MTB, self).__init__()
        self.block = nn.Sequential()
        self.out_channels = in_channels
        self.cnn_num = cnn_num
        if self.cnn_num == 2:
            for i in range(self.cnn_num):
                self.block.add_sublayer('conv_{}'.format(i), nn.Conv2D(
                    in_channels = in_channels if i == 0 else 32*(2**(i-1)), 
                    out_channels = 32*(2**i), 
                    kernel_size = 3, 
                    stride = 2, 
                    padding=1))
                self.block.add_sublayer('relu_{}'.format(i), nn.ReLU())
                self.block.add_sublayer('bn_{}'.format(i), nn.BatchNorm2D(32*(2**i)))

    def forward(self, images):
        
        x = self.block(images)
        if self.cnn_num == 2:
            # (b, w, h, c)
            x = x.transpose([0, 3, 2, 1])
            x_shape = x.shape
            x = x.reshape([x_shape[0], x_shape[1], x_shape[2] * x_shape[3]])
        return x
