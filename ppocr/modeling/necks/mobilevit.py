import paddle
import paddle.nn as nn


def _init_weights_linear():
    weight_attr = paddle.ParamAttr(initializer=nn.initializer.TruncatedNormal(
        std=.02))
    bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.0))
    return weight_attr, bias_attr


def _init_weights_layernorm():
    weight_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(1.0))
    bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.0))
    return weight_attr, bias_attr


# DONE
class ConvNormAct(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 groups=1):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.KaimingUniform()),
            bias_attr=bias_attr)
        self.norm = nn.BatchNorm2D(out_channels)
        self.act = nn.Silu()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


# DONE
class Identity(nn.Layer):
    """ Identity layer"""

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs


#DONE
class Mlp(nn.Layer):
    def __init__(self, embed_dim, mlp_ratio, dropout=0.):
        super().__init__()
        w_attr_1, b_attr_1 = _init_weights_linear()
        self.fc1 = nn.Linear(
            embed_dim,
            int(embed_dim * mlp_ratio),
            weight_attr=w_attr_1,
            bias_attr=b_attr_1)

        w_attr_2, b_attr_2 = _init_weights_linear()
        self.fc2 = nn.Linear(
            int(embed_dim * mlp_ratio),
            embed_dim,
            weight_attr=w_attr_2,
            bias_attr=b_attr_2)

        self.act = nn.Silu()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class Attention(nn.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.attn_head_dim = int(embed_dim / self.num_heads)
        self.all_head_dim = self.attn_head_dim * self.num_heads

        w_attr_1, b_attr_1 = _init_weights_linear()
        self.qkv = nn.Linear(
            embed_dim,
            self.all_head_dim * 3,  # weights for q, k, v
            weight_attr=w_attr_1,
            bias_attr=b_attr_1 if qkv_bias else False)

        self.scales = self.attn_head_dim**-0.5

        w_attr_2, b_attr_2 = _init_weights_linear()
        self.proj = nn.Linear(
            embed_dim, embed_dim, weight_attr=w_attr_2, bias_attr=b_attr_2)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)

    def transpose_multihead(self, x):
        # in_shape: [batch_size, P, N, hd]
        B, P, N, d = x.shape
        x = x.reshape([0, P, N, self.num_heads, d // self.num_heads])
        x = x.transpose([0, 1, 3, 2, 4])
        # out_shape: [batch_size, P, num_heads, N, d]
        return x

    def forward(self, x):
        # [B, 2x2, 256, 96]: [B, P, N, d]
        qkv = self.qkv(x).chunk(3, axis=-1)
        q, k, v = map(self.transpose_multihead, qkv)

        attn = paddle.matmul(q, k, transpose_y=True)
        attn = attn * self.scales
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)
        # [batch_size, P, num_heads, N, N]

        z = paddle.matmul(attn, v)
        # [batch_size, P, num_heads, N, d]
        z = z.transpose([0, 1, 3, 2, 4])
        B, P, N, H, D = z.shape
        z = z.reshape([0, P, N, H * D])
        z = self.proj(z)
        z = self.proj_dropout(z)
        return z


# DONE
class EncoderLayer(nn.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads=8,
                 qkv_bias=True,
                 mlp_ratio=2.0,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()
        w_attr_1, b_attr_1 = _init_weights_layernorm()
        w_attr_2, b_attr_2 = _init_weights_layernorm()

        self.attn_norm = nn.LayerNorm(
            embed_dim, weight_attr=w_attr_1, bias_attr=b_attr_1)
        self.attn = Attention(embed_dim, num_heads, qkv_bias, attention_dropout,
                              dropout)
        self.drop_path = DropPath(droppath) if droppath > 0. else Identity()
        self.mlp_norm = nn.LayerNorm(
            embed_dim, weight_attr=w_attr_2, bias_attr=b_attr_2)
        self.mlp = Mlp(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        h = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x = self.drop_path(x)
        x = h + x

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + h
        return x


# DONE
class Transformer(nn.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 depth,
                 qkv_bias=True,
                 mlp_ratio=2.0,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()
        depth_decay = [x.item() for x in paddle.linspace(0, droppath, depth)]

        layer_list = []
        for i in range(depth):
            layer_list.append(
                EncoderLayer(embed_dim, num_heads, qkv_bias, mlp_ratio, dropout,
                             attention_dropout, droppath))
        self.layers = nn.LayerList(layer_list)

        w_attr_1, b_attr_1 = _init_weights_layernorm()
        self.norm = nn.LayerNorm(
            embed_dim, weight_attr=w_attr_1, bias_attr=b_attr_1, epsilon=1e-6)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        out = self.norm(x)
        return out


# DONE
class MobileViTBlock(nn.Layer):
    def __init__(self,
                 dim,
                 hidden_dim,
                 depth,
                 num_heads=8,
                 qkv_bias=True,
                 mlp_ratio=2.0,
                 dropout=0.1,
                 attention_dropout=0.1,
                 droppath=0.,
                 patch_size=(1, 1)):
        super().__init__()
        self.patch_h, self.patch_w = patch_size

        # local representations
        self.conv1 = ConvNormAct(dim, dim // 8, padding=1)
        self.conv2 = ConvNormAct(dim // 8, hidden_dim, kernel_size=1)

        # global representations
        self.transformer = Transformer(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            depth=depth,
            qkv_bias=qkv_bias,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            droppath=droppath)

        # fusion
        self.conv3 = ConvNormAct(hidden_dim, dim, kernel_size=1)
        # last conv-nxn, the input is concat of input tensor and conv3 output tensor
        self.conv4 = ConvNormAct(2 * dim, dim // 8, padding=1)

    def forward(self, x):
        h = x
        x = self.conv1(x)
        x = self.conv2(x)

        # [B, 96, 32, 32]

        B, C, H, W = x.shape
        # print(x.shape)
        # x.reshape([B, C, H//self.patch_h, self.patch_w, W//self.patch_w, self.patch_w])
        # [B, C, H, 1, W, 1]
        x = paddle.unsqueeze(x, axis=[3, 5])
        # [4, 96, 16, 2, 16, 2]
        x = paddle.transpose(x, perm=[0, 1, 3, 5, 2, 4])
        # [4, 96, 2, 2, 16, 16]
        x = x.reshape([0, C, (self.patch_h * self.patch_w),
                       H * W])  #[B, C, ws ** 2, n_windows ** 2]
        x = x.transpose([0, 2, 3, 1])  #[B, ws ** 2, n_windows ** 2, C]
        # [4, 4, 256, 96]
        x = self.transformer(x)
        x = x.reshape([
            0, self.patch_h, self.patch_w, H // self.patch_h, W // self.patch_w,
            C
        ])
        x = x.transpose([0, 5, 3, 1, 4, 2])
        x = x.reshape([0, C, H, W])
        x = self.conv3(x)
        x = paddle.concat((h, x), axis=1)
        x = self.conv4(x)
        return x
