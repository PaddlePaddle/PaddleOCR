# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

# Modified from
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# Copyright 2020 Ross Wightman, Apache-2.0 License

import paddle
import itertools
#import utils
import math
import warnings
import paddle.nn.functional as F
from paddle.nn.initializer import TruncatedNormal, Constant

#from timm.models.vision_transformer import trunc_normal_
#from timm.models.registry import register_model

specification = {
    'LeViT_128S': {
        'C': '128_256_384',
        'D': 16,
        'N': '4_6_8',
        'X': '2_3_4',
        'drop_path': 0,
        'weights':
        'https://dl.fbaipublicfiles.com/LeViT/LeViT-128S-96703c44.pth'
    },
    'LeViT_128': {
        'C': '128_256_384',
        'D': 16,
        'N': '4_8_12',
        'X': '4_4_4',
        'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-128-b88c2750.pth'
    },
    'LeViT_192': {
        'C': '192_288_384',
        'D': 32,
        'N': '3_5_6',
        'X': '4_4_4',
        'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-192-92712e41.pth'
    },
    'LeViT_256': {
        'C': '256_384_512',
        'D': 32,
        'N': '4_6_8',
        'X': '4_4_4',
        'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-256-13b5763e.pth'
    },
    'LeViT_384': {
        'C': '384_512_768',
        'D': 32,
        'N': '6_9_12',
        'X': '4_4_4',
        'drop_path': 0.1,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-384-9bdaf2e2.pth'
    },
}

__all__ = [specification.keys()]

trunc_normal_ = TruncatedNormal(std=.02)
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


#@register_model
def LeViT_128S(class_dim=1000, distillation=True, pretrained=False, fuse=False):
    return model_factory(
        **specification['LeViT_128S'],
        class_dim=class_dim,
        distillation=distillation,
        pretrained=pretrained,
        fuse=fuse)


#@register_model
def LeViT_128(class_dim=1000, distillation=True, pretrained=False, fuse=False):
    return model_factory(
        **specification['LeViT_128'],
        class_dim=class_dim,
        distillation=distillation,
        pretrained=pretrained,
        fuse=fuse)


#@register_model
def LeViT_192(class_dim=1000, distillation=True, pretrained=False, fuse=False):
    return model_factory(
        **specification['LeViT_192'],
        class_dim=class_dim,
        distillation=distillation,
        pretrained=pretrained,
        fuse=fuse)


#@register_model
def LeViT_256(class_dim=1000, distillation=False, pretrained=False, fuse=False):
    return model_factory(
        **specification['LeViT_256'],
        class_dim=class_dim,
        distillation=distillation,
        pretrained=pretrained,
        fuse=fuse)


#@register_model
def LeViT_384(class_dim=1000, distillation=True, pretrained=False, fuse=False):
    return model_factory(
        **specification['LeViT_384'],
        class_dim=class_dim,
        distillation=distillation,
        pretrained=pretrained,
        fuse=fuse)


FLOPS_COUNTER = 0


class Conv2d_BN(paddle.nn.Sequential):
    def __init__(self,
                 a,
                 b,
                 ks=1,
                 stride=1,
                 pad=0,
                 dilation=1,
                 groups=1,
                 bn_weight_init=1,
                 resolution=-10000):
        super().__init__()
        self.add_sublayer(
            'c',
            paddle.nn.Conv2D(
                a, b, ks, stride, pad, dilation, groups, bias_attr=False))
        bn = paddle.nn.BatchNorm2D(b)
        ones_(bn.weight)
        zeros_(bn.bias)
        self.add_sublayer('bn', bn)

        global FLOPS_COUNTER
        output_points = (
            (resolution + 2 * pad - dilation * (ks - 1) - 1) // stride + 1)**2
        FLOPS_COUNTER += a * b * output_points * (ks**2)

    @paddle.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = paddle.nn.Conv2D(
            w.size(1),
            w.size(0),
            w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Linear_BN(paddle.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1, resolution=-100000):
        super().__init__()
        self.add_sublayer('c', paddle.nn.Linear(a, b, bias_attr=False))
        bn = paddle.nn.BatchNorm1D(b)
        ones_(bn.weight)
        zeros_(bn.bias)
        self.add_sublayer('bn', bn)

        global FLOPS_COUNTER
        output_points = resolution**2
        FLOPS_COUNTER += a * b * output_points

    @paddle.no_grad()
    def fuse(self):
        l, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = paddle.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        l, bn = self._sub_layers.values()
        x = l(x)
        return paddle.reshape(bn(x.flatten(0, 1)), x.shape)


class BN_Linear(paddle.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_sublayer('bn', paddle.nn.BatchNorm1D(a))
        l = paddle.nn.Linear(a, b, bias_attr=bias)
        trunc_normal_(l.weight)
        if bias:
            zeros_(l.bias)
        self.add_sublayer('l', l)
        global FLOPS_COUNTER
        FLOPS_COUNTER += a * b

    @paddle.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @self.l.weight.T
        else:
            b = (l.weight @b[:, None]).view(-1) + self.l.bias
        m = paddle.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


def b16(n, activation, resolution=224):
    return paddle.nn.Sequential(
        Conv2d_BN(
            3, n // 8, 3, 2, 1, resolution=resolution),
        activation(),
        Conv2d_BN(
            n // 8, n // 4, 3, 2, 1, resolution=resolution // 2),
        activation(),
        Conv2d_BN(
            n // 4, n // 2, 3, 2, 1, resolution=resolution // 4),
        activation(),
        Conv2d_BN(
            n // 2, n, 3, 2, 1, resolution=resolution // 8))


class Residual(paddle.nn.Layer):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * paddle.rand(
                x.size(0), 1, 1,
                device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class Attention(paddle.nn.Layer):
    def __init__(self,
                 dim,
                 key_dim,
                 num_heads=8,
                 attn_ratio=4,
                 activation=None,
                 resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        self.h = self.dh + nh_kd * 2
        self.qkv = Linear_BN(dim, self.h, resolution=resolution)
        self.proj = paddle.nn.Sequential(
            activation(),
            Linear_BN(
                self.dh, dim, bn_weight_init=0, resolution=resolution))
        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = self.create_parameter(
            shape=(num_heads, len(attention_offsets)),
            default_initializer=zeros_)
        tensor_idxs = paddle.to_tensor(idxs, dtype='int64')
        self.register_buffer('attention_bias_idxs',
                             paddle.reshape(tensor_idxs, [N, N]))

        global FLOPS_COUNTER
        #queries * keys
        FLOPS_COUNTER += num_heads * (resolution**4) * key_dim
        # softmax
        FLOPS_COUNTER += num_heads * (resolution**4)
        #attention * v
        FLOPS_COUNTER += num_heads * self.d * (resolution**4)

    @paddle.no_grad()
    def train(self, mode=True):
        if mode:
            super().train()
        else:
            super().eval()
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            gather_list = []
            attention_bias_t = paddle.transpose(self.attention_biases, (1, 0))
            for idx in self.attention_bias_idxs:
                gather = paddle.gather(attention_bias_t, idx)
                gather_list.append(gather)
            attention_biases = paddle.transpose(
                paddle.concat(gather_list), (1, 0)).reshape(
                    (0, self.attention_bias_idxs.shape[0],
                     self.attention_bias_idxs.shape[1]))
            self.ab = attention_biases
            #self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        self.training = True
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = paddle.reshape(qkv,
                             [B, N, self.num_heads, self.h // self.num_heads])
        q, k, v = paddle.split(
            qkv, [self.key_dim, self.key_dim, self.d], axis=3)
        q = paddle.transpose(q, perm=[0, 2, 1, 3])
        k = paddle.transpose(k, perm=[0, 2, 1, 3])
        v = paddle.transpose(v, perm=[0, 2, 1, 3])
        k_transpose = paddle.transpose(k, perm=[0, 1, 3, 2])

        if self.training:
            gather_list = []
            attention_bias_t = paddle.transpose(self.attention_biases, (1, 0))
            for idx in self.attention_bias_idxs:
                gather = paddle.gather(attention_bias_t, idx)
                gather_list.append(gather)
            attention_biases = paddle.transpose(
                paddle.concat(gather_list), (1, 0)).reshape(
                    (0, self.attention_bias_idxs.shape[0],
                     self.attention_bias_idxs.shape[1]))
        else:
            attention_biases = self.ab
        #np_ = paddle.to_tensor(self.attention_biases.numpy()[:, self.attention_bias_idxs.numpy()])
        #print(self.attention_bias_idxs.shape)
        #print(attention_biases.shape)
        #print(np_.shape)
        #print(np_.equal(attention_biases))
        #exit()

        attn = ((q @k_transpose) * self.scale + attention_biases)
        attn = F.softmax(attn)
        x = paddle.transpose(attn @v, perm=[0, 2, 1, 3])
        x = paddle.reshape(x, [B, N, self.dh])
        x = self.proj(x)
        return x


class Subsample(paddle.nn.Layer):
    def __init__(self, stride, resolution):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        x = paddle.reshape(x, [B, self.resolution, self.resolution,
                               C])[:, ::self.stride, ::self.stride]
        x = paddle.reshape(x, [B, -1, C])
        return x


class AttentionSubsample(paddle.nn.Layer):
    def __init__(self,
                 in_dim,
                 out_dim,
                 key_dim,
                 num_heads=8,
                 attn_ratio=2,
                 activation=None,
                 stride=2,
                 resolution=14,
                 resolution_=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.resolution_ = resolution_
        self.resolution_2 = resolution_**2
        self.training = True
        h = self.dh + nh_kd
        self.kv = Linear_BN(in_dim, h, resolution=resolution)

        self.q = paddle.nn.Sequential(
            Subsample(stride, resolution),
            Linear_BN(
                in_dim, nh_kd, resolution=resolution_))
        self.proj = paddle.nn.Sequential(
            activation(), Linear_BN(
                self.dh, out_dim, resolution=resolution_))

        self.stride = stride
        self.resolution = resolution
        points = list(itertools.product(range(resolution), range(resolution)))
        points_ = list(
            itertools.product(range(resolution_), range(resolution_)))

        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        i = 0
        j = 0
        for p1 in points_:
            i += 1
            for p2 in points:
                j += 1
                size = 1
                offset = (abs(p1[0] * stride - p2[0] + (size - 1) / 2),
                          abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = self.create_parameter(
            shape=(num_heads, len(attention_offsets)),
            default_initializer=zeros_)

        tensor_idxs_ = paddle.to_tensor(idxs, dtype='int64')
        self.register_buffer('attention_bias_idxs',
                             paddle.reshape(tensor_idxs_, [N_, N]))

        global FLOPS_COUNTER
        #queries * keys
        FLOPS_COUNTER += num_heads * \
            (resolution**2) * (resolution_**2) * key_dim
        # softmax
        FLOPS_COUNTER += num_heads * (resolution**2) * (resolution_**2)
        #attention * v
        FLOPS_COUNTER += num_heads * \
            (resolution**2) * (resolution_**2) * self.d

    @paddle.no_grad()
    def train(self, mode=True):
        if mode:
            super().train()
        else:
            super().eval()
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            gather_list = []
            attention_bias_t = paddle.transpose(self.attention_biases, (1, 0))
            for idx in self.attention_bias_idxs:
                gather = paddle.gather(attention_bias_t, idx)
                gather_list.append(gather)
            attention_biases = paddle.transpose(
                paddle.concat(gather_list), (1, 0)).reshape(
                    (0, self.attention_bias_idxs.shape[0],
                     self.attention_bias_idxs.shape[1]))
            self.ab = attention_biases
            #self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        self.training = True
        B, N, C = x.shape
        kv = self.kv(x)
        kv = paddle.reshape(kv, [B, N, self.num_heads, -1])
        k, v = paddle.split(kv, [self.key_dim, self.d], axis=3)
        k = paddle.transpose(k, perm=[0, 2, 1, 3])  # BHNC
        v = paddle.transpose(v, perm=[0, 2, 1, 3])
        q = paddle.reshape(
            self.q(x), [B, self.resolution_2, self.num_heads, self.key_dim])
        q = paddle.transpose(q, perm=[0, 2, 1, 3])

        if self.training:
            gather_list = []
            attention_bias_t = paddle.transpose(self.attention_biases, (1, 0))
            for idx in self.attention_bias_idxs:
                gather = paddle.gather(attention_bias_t, idx)
                gather_list.append(gather)
            attention_biases = paddle.transpose(
                paddle.concat(gather_list), (1, 0)).reshape(
                    (0, self.attention_bias_idxs.shape[0],
                     self.attention_bias_idxs.shape[1]))
        else:
            attention_biases = self.ab

        attn = (q @paddle.transpose(
            k, perm=[0, 1, 3, 2])) * self.scale + attention_biases
        attn = F.softmax(attn)

        x = paddle.reshape(
            paddle.transpose(
                (attn @v), perm=[0, 2, 1, 3]), [B, -1, self.dh])
        x = self.proj(x)
        return x


class LeViT(paddle.nn.Layer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 class_dim=1000,
                 embed_dim=[192],
                 key_dim=[64],
                 depth=[12],
                 num_heads=[3],
                 attn_ratio=[2],
                 mlp_ratio=[2],
                 hybrid_backbone=None,
                 down_ops=[],
                 attention_activation=paddle.nn.Hardswish,
                 mlp_activation=paddle.nn.Hardswish,
                 distillation=True,
                 drop_path=0):
        super().__init__()
        global FLOPS_COUNTER

        self.class_dim = class_dim
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        self.distillation = distillation

        self.patch_embed = hybrid_backbone

        self.blocks = []
        down_ops.append([''])
        resolution = img_size // patch_size
        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, mlp_ratio,
                    down_ops)):
            for _ in range(dpth):
                self.blocks.append(
                    Residual(
                        Attention(
                            ed,
                            kd,
                            nh,
                            attn_ratio=ar,
                            activation=attention_activation,
                            resolution=resolution, ),
                        drop_path))
                if mr > 0:
                    h = int(ed * mr)
                    self.blocks.append(
                        Residual(
                            paddle.nn.Sequential(
                                Linear_BN(
                                    ed, h, resolution=resolution),
                                mlp_activation(),
                                Linear_BN(
                                    h,
                                    ed,
                                    bn_weight_init=0,
                                    resolution=resolution), ),
                            drop_path))
            if do[0] == 'Subsample':
                #('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
                resolution_ = (resolution - 1) // do[5] + 1
                self.blocks.append(
                    AttentionSubsample(
                        *embed_dim[i:i + 2],
                        key_dim=do[1],
                        num_heads=do[2],
                        attn_ratio=do[3],
                        activation=attention_activation,
                        stride=do[5],
                        resolution=resolution,
                        resolution_=resolution_))
                resolution = resolution_
                if do[4] > 0:  # mlp_ratio
                    h = int(embed_dim[i + 1] * do[4])
                    self.blocks.append(
                        Residual(
                            paddle.nn.Sequential(
                                Linear_BN(
                                    embed_dim[i + 1], h, resolution=resolution),
                                mlp_activation(),
                                Linear_BN(
                                    h,
                                    embed_dim[i + 1],
                                    bn_weight_init=0,
                                    resolution=resolution), ),
                            drop_path))
        self.blocks = paddle.nn.Sequential(*self.blocks)

        # Classifier head
        self.head = BN_Linear(
            embed_dim[-1], class_dim) if class_dim > 0 else paddle.nn.Identity()
        if distillation:
            self.head_dist = BN_Linear(
                embed_dim[-1],
                class_dim) if class_dim > 0 else paddle.nn.Identity()

        self.FLOPS = FLOPS_COUNTER
        FLOPS_COUNTER = 0

    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = paddle.transpose(x, perm=[0, 2, 1])
        x = self.blocks(x)
        x = x.mean(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x


def model_factory(C, D, X, N, drop_path, weights, class_dim, distillation,
                  pretrained, fuse):
    embed_dim = [int(x) for x in C.split('_')]
    num_heads = [int(x) for x in N.split('_')]
    depth = [int(x) for x in X.split('_')]
    act = paddle.nn.Hardswish
    model = LeViT(
        patch_size=16,
        embed_dim=embed_dim,
        num_heads=num_heads,
        key_dim=[D] * 3,
        depth=depth,
        attn_ratio=[2, 2, 2],
        mlp_ratio=[2, 2, 2],
        down_ops=[
            #('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ['Subsample', D, embed_dim[0] // D, 4, 2, 2],
            ['Subsample', D, embed_dim[1] // D, 4, 2, 2],
        ],
        attention_activation=act,
        mlp_activation=act,
        hybrid_backbone=b16(embed_dim[0], activation=act),
        class_dim=class_dim,
        drop_path=drop_path,
        distillation=distillation)
    #     if pretrained:
    #         checkpoint = torch.hub.load_state_dict_from_url(
    #             weights, map_location='cpu')
    #         model.load_state_dict(checkpoint['model'])
    if fuse:
        utils.replace_batchnorm(model)

    return model


if __name__ == '__main__':
    '''
    import torch
    checkpoint = torch.load('../LeViT/pretrained256.pth')
    torch_dict = checkpoint['net']
    paddle_dict = {}
    fc_names = ["c.weight", "l.weight", "qkv.weight", "fc1.weight", "fc2.weight", "downsample.reduction.weight", "head.weight", "attn.proj.weight"]
    rename_dict = {"running_mean": "_mean", "running_var": "_variance"}
    range_tuple = (0, 502)
    idx = 0
    for key in torch_dict:
        idx += 1
        weight = torch_dict[key].cpu().numpy()
        flag = [i in key for i in fc_names]
        if any(flag):
            if "emb" not in key:
                print("weight {} need to be trans".format(key))
                weight = weight.transpose()
        key = key.replace("running_mean", "_mean")
        key = key.replace("running_var", "_variance")
        paddle_dict[key]=weight
    '''
    import numpy as np
    net = globals()['LeViT_256'](fuse=False,
                                 pretrained=False,
                                 distillation=False)
    load_layer_state_dict = paddle.load(
        "./LeViT_256_official_nodistillation_paddle.pdparams")
    #net.set_state_dict(paddle_dict)
    net.set_state_dict(load_layer_state_dict)
    net.eval()
    #paddle.save(net.state_dict(), "./LeViT_256_official_paddle.pdparams")
    #model = paddle.jit.to_static(net,input_spec=[paddle.static.InputSpec(shape=[None, 3, 224, 224], dtype='float32')])
    #paddle.jit.save(model, "./LeViT_256_official_inference/inference")
    #exit()
    np.random.seed(123)
    img = np.random.rand(1, 3, 224, 224).astype('float32')
    img = paddle.to_tensor(img)
    outputs = net(img).numpy()
    print(outputs[0][:10])
    #print(outputs.shape)
