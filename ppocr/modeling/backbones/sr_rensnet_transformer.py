import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math, copy
import numpy as np



#from ppocr.modeling.heads.multiheadAttention import MultiheadAttention

# stroke-level alphabet
alphabet = '0123456789'

def get_alphabet_len():
    return len(alphabet)


# def subsequent_mask(size):
#     attn_shape = (1, size, size)
#     subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
#     return paddle.from_numpy(subsequent_mask) == 0


def subsequent_mask(size):
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = paddle.ones([1, size, size], dtype='float32')
    mask_inf = paddle.triu(
        paddle.full(
            shape=[1, size, size], dtype='float32', fill_value='-inf'),
        diagonal=1)
    mask = mask + mask_inf
    padding_mask = paddle.equal(mask, paddle.to_tensor(1, dtype=mask.dtype))
    return padding_mask



def clones(module, N):
    return nn.LayerList([copy.deepcopy(module) for _ in range(N)])

def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)

def attention(query, key, value, mask=None, dropout=None, attention_map=None):
    d_k = query.shape[-1]
    scores = paddle.matmul(query, paddle.transpose(key, [0,1,3,2])) / math.sqrt(d_k)
    
    if mask is not None:
        scores = masked_fill(scores, mask == 0, float('-inf'))
    else:
        pass

    p_attn = F.softmax(scores, axis=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return paddle.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Layer):
    def __init__(self, h, d_model, dropout=0.0, compress_attention=False):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout, mode="downscale_in_infer")
        self.compress_attention = compress_attention
        self.compress_attention_linear = nn.Linear(h, 1)

    def forward(self, query, key, value, mask=None, attention_map=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.shape[0]
        # print("query:{}, key:{}, value:{}".format(np.sum(query.numpy()), np.sum(key.numpy()), np.sum(value.numpy())))
        # print("============ befor ==========")

        query, key, value = \
            [paddle.transpose(l(x).reshape([nbatches, -1, self.h, self.d_k]), [0,2,1,3])
             for l, x in zip(self.linears, (query, key, value))]
        # print("query:{}, key:{}, value:{}".format(np.sum(query.numpy()), np.sum(key.numpy()), np.sum(value.numpy())))

        x, attention_map = attention(query, key, value, mask=mask,
                                     dropout=self.dropout, attention_map=attention_map)

        x = paddle.reshape(paddle.transpose(x, [0, 2, 1, 3]), [nbatches, -1, self.h*self.d_k])

        return self.linears[-1](x), attention_map


class ResNet(nn.Layer):

    def __init__(self, num_in, block, layers):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2D(num_in, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(64,use_global_stats=True)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2D((2, 2), (2, 2))

        self.conv2 = nn.Conv2D(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(128,use_global_stats=True)
        self.relu2 = nn.ReLU()

        self.layer1_pool = nn.MaxPool2D((2, 2), (2, 2))
        self.layer1 = self._make_layer(block, 128, 256, layers[0])
        self.layer1_conv = nn.Conv2D(256, 256, 3, 1, 1)
        self.layer1_bn = nn.BatchNorm2D(256,use_global_stats=True)
        self.layer1_relu = nn.ReLU()

        self.layer2_pool = nn.MaxPool2D((2, 2), (2, 2))
        self.layer2 = self._make_layer(block, 256, 256, layers[1])
        self.layer2_conv = nn.Conv2D(256, 256, 3, 1, 1)
        self.layer2_bn = nn.BatchNorm2D(256, use_global_stats=True)
        self.layer2_relu = nn.ReLU()

        self.layer3_pool = nn.MaxPool2D((2, 2), (2, 2))
        self.layer3 = self._make_layer(block, 256, 512, layers[2])
        self.layer3_conv = nn.Conv2D(512, 512, 3, 1, 1)
        self.layer3_bn = nn.BatchNorm2D(512, use_global_stats=True)
        self.layer3_relu = nn.ReLU()

        self.layer4_pool = nn.MaxPool2D((2, 2), (2, 2))
        self.layer4 = self._make_layer(block, 512, 512, layers[3])
        self.layer4_conv2 = nn.Conv2D(512, 1024, 3, 1, 1)
        self.layer4_conv2_bn = nn.BatchNorm2D(1024, use_global_stats=True)
        self.layer4_conv2_relu = nn.ReLU()

    def _make_layer(self, block, inplanes, planes, blocks):

        if inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2D(inplanes, planes, 3, 1, 1),
                nn.BatchNorm2D(planes, use_global_stats=True), )
        else:
            downsample = None
        layers = []
        layers.append(block(inplanes, planes, downsample))
        for i in range(1, blocks):
            layers.append(block(planes, planes, downsample=None))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print("input x:", np.sum(x.numpy()))
        x = self.conv1(x)
        # print("=====")
        # print("conv1 weight:", np.sum(self.conv1.weight.numpy()))
        # print("=====")
        # print("x shape:", x.shape)
        # print("bn weights:", np.sum(self.bn1.weight.numpy()))
        # print("bn bias:", np.sum(self.bn1.bias.numpy()))
        # print("bn mean:", np.sum(self.bn1._mean.numpy()))
        # print("bn var:", np.sum(self.bn1._variance.numpy()))
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)


        x = self.layer1_pool(x)
        x = self.layer1(x)
        x = self.layer1_conv(x)
        x = self.layer1_bn(x)
        x = self.layer1_relu(x)


        # x = self.layer2_pool(x)
        x = self.layer2(x)
        x = self.layer2_conv(x)
        x = self.layer2_bn(x)
        x = self.layer2_relu(x)


        # x = self.layer3_pool(x)
        x = self.layer3(x)
        x = self.layer3_conv(x)
        x = self.layer3_bn(x)
        x = self.layer3_relu(x)


        # x = self.layer4_pool(x)
        x = self.layer4(x)
        x = self.layer4_conv2(x)
        x = self.layer4_conv2_bn(x)
        x = self.layer4_conv2_relu(x)

        return x


class Bottleneck(nn.Layer):

    def __init__(self, input_dim):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(input_dim, input_dim, 1)
        self.bn1 = nn.BatchNorm2D(input_dim, use_global_stats=True)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2D(input_dim, input_dim, 3, 1, 1)
        self.bn2 = nn.BatchNorm2D(input_dim, use_global_stats=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.se(out)

        out += residual
        out = self.relu(out)

        return out


class PositionalEncoding(nn.Layer):
    "Implement the PE function."

    def __init__(self, dropout,  dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout,mode="downscale_in_infer")

        pe = paddle.zeros([max_len, dim])
        position = paddle.arange(0, max_len, dtype=paddle.float32).unsqueeze(1)
        div_term = paddle.exp(
            paddle.arange(0, dim, 2).astype('float32') *
            (-math.log(10000.0) / dim))
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        pe = paddle.unsqueeze(pe, 0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :paddle.shape(x)[1]]
        return self.dropout(x)


# class LayerNorm(nn.Layer):
#     "Construct a layernorm module (See citation for details)."

#     def __init__(self, features, eps=1e-6):
#         super(LayerNorm, self).__init__()
#         self.a_2 = Paddle.nn.initParameter(paddle.ones(features))
#         self.b_2 = nn.Parameter(paddle.zeros(features))
#         self.eps = eps

#     def forward(self, x):
#         mean = x.mean(-1, keepdim=True)
#         std = x.std(-1, keepdim=True)
#         return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Layer):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout,mode="downscale_in_infer")

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Generator(nn.Layer):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.relu = nn.ReLU()

    def forward(self, x):
        # return F.softmax(self.proj(x))
        out = self.proj(x)
        return out


class Embeddings(nn.Layer):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        embed = self.lut(x) * math.sqrt(self.d_model)
        # print("embed",embed)
        # embed = self.lut(x)
        # print(embed.requires_grad)
        return embed


class LayerNorm(nn.Layer):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # print(features)
        self.a_2 = self.create_parameter(
            shape=[features],
            default_initializer=paddle.nn.initializer.Constant(1.0))
        self.b_2 = self.create_parameter(
            shape=[features],
            default_initializer=paddle.nn.initializer.Constant(0.0))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Decoder(nn.Layer):

    def __init__(self):
        super(Decoder, self).__init__()

        self.mask_multihead = MultiHeadedAttention(h=16, d_model=1024, dropout=0.0)
        self.mul_layernorm1 = LayerNorm(1024)

        self.multihead = MultiHeadedAttention(h=16, d_model=1024, dropout=0.0)
        self.mul_layernorm2 = LayerNorm(1024)

        self.pff = PositionwiseFeedForward(1024, 2048)
        self.mul_layernorm3 = LayerNorm(1024)

    def forward(self, text, conv_feature, attention_map=None):
        text_max_length = text.shape[1]
        mask = subsequent_mask(text_max_length)
        result = text
        # print("result:", np.sum(result.numpy()))
        # print("mask:", np.sum(mask.numpy()))
        # print("tmp:", np.sum(tmp.numpy()))
        result = self.mul_layernorm1(result + self.mask_multihead(result, result, result, mask=mask)[0])
        # print("result:", np.sum(result.numpy()))
        # print("layer weight:", self.mul_layernorm1.a_2.numpy())
        # print("my layernorm:", np.sum(result.numpy()))
        b, c, h, w = conv_feature.shape
        conv_feature = paddle.transpose(conv_feature.reshape([b, c, h * w]), [0, 2, 1])
        # print("conv feature:", np.sum(conv_feature.numpy()))
        word_image_align, attention_map = self.multihead(result, conv_feature, conv_feature, mask=None, attention_map=attention_map)
        result = self.mul_layernorm2(result + word_image_align)

        result = self.mul_layernorm3(result + self.pff(result))

        return result, attention_map


class BasicBlock(nn.Layer):

    def __init__(self, inplanes, planes, downsample):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(planes, use_global_stats=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(planes, use_global_stats=True)
        self.downsample = downsample
        # self.se = SELayer(planes, 16)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.se(out)

        if self.downsample != None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class Encoder(nn.Layer):

    def __init__(self):
        super(Encoder, self).__init__()
        self.cnn = ResNet(num_in=1, block=BasicBlock, layers=[1, 2, 5, 3])

    def forward(self, input):
        conv_result = self.cnn(input)
        return conv_result


class Transformer(nn.Layer):

    def __init__(self,
                 in_channels=1):
        super(Transformer, self).__init__()

        word_n_class = get_alphabet_len()
        self.embedding_word_with_upperword = Embeddings(512, word_n_class)
        self.pe = PositionalEncoding(dim=512, dropout=0.0, max_len=5000)

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.generator_word_with_upperword = Generator(1024, word_n_class)


        for p in self.parameters():
            if p.dim() > 1:
                nn.initializer.XavierNormal(p)

    def forward(self, image, text_length, text_input, test=False, attention_map=None):
        if image.shape[1] == 3:
            R = image[:, 0:1, :, :]
            G = image[:, 1:2, :, :]
            B = image[:, 2:3, :, :]
            image = 0.299 * R + 0.587 * G + 0.114 * B
        
        # print("image shape:", image.shape)

        conv_feature = self.encoder(image) # batch, 1024, 8, 32
        # print("conv feature:", np.sum(conv_feature.numpy()))
        max_length = max(text_length)
        text_input = text_input[:,:max_length]
        # print("length:", text_length)
        # print("input tensor:", text_input)
        text_embedding = self.embedding_word_with_upperword(text_input) # batch, text_max_length, 512
        # print("text_embedding:", np.sum(text_embedding.numpy()))
        postion_embedding = self.pe(paddle.zeros(text_embedding.shape)) # batch, text_max_length, 512
        # print("postion_embedding:", np.sum(postion_embedding.numpy()))
        text_input_with_pe = paddle.concat([text_embedding, postion_embedding], 2) # batch, text_max_length, 1024
        batch, seq_len, _ = text_input_with_pe.shape

        # print("text_input_with_pe:", np.sum(text_input_with_pe.numpy()))
        text_input_with_pe, word_attention_map = self.decoder(text_input_with_pe, conv_feature)

        # print("text_input_with_pe:", np.sum(text_input_with_pe.numpy()))
        # print("attention map:", np.sum(word_attention_map.numpy()))
        word_decoder_result = self.generator_word_with_upperword(text_input_with_pe)
        correct_list = []


        if not test:
            total_length = paddle.sum(text_length)
            probs_res = paddle.zeros([total_length, get_alphabet_len()])
            start = 0

            for index, length in enumerate(text_length):
                length = int(length.numpy())
                probs_res[start:start + length, :] = word_decoder_result[index, 0:0 + length, :]
                # if (paddle.equal_all(paddle.argmax(probs_res[start:start + length, :], axis=1)[:-1] , text_input[index][1:length])):
                #     correct_list.append(True)
                # else:
                #     correct_list.append(False)
                start = start + length

            # return probs_res, word_attention_map, correct_list
            # print("probs res:", np.sum(probs_res.numpy()))
            return probs_res, word_attention_map, correct_list # there is a bug
        else:
            return word_decoder_result


if __name__ == '__main__':
    #image = torch.Tensor(32,1,32,128).cuda()
    """
    image = paddle.rand([32,1,32,128])
    model = Encoder()
    output = model(image)
    print("output shape:", output.shape)

    word_n_class = get_alphabet_len()
    embedding_word = Embeddings(512, word_n_class)
    # word = torch.Tensor([[1,2,3],[4,5,6]]) # 2,3 -> 2, 3, 512
    word = paddle.to_tensor([[1,2,3],[4,5,6]])
    embedding = embedding_word(word)
    print("embedding shape:", embedding.shape)

    #text_input = torch.Tensor(32,10,1024)
    text_input = paddle.rand([32,10,1024])
    image_input = paddle.rand([32,1024,8,32])
    # image_input = torch.Tensor(32,1024,8,32)
    decoder = Decoder()
    decoder_output = decoder(text_input, image_input)
    print("decode out [0]:", decoder_output[0].shape)
    print("decode out [1]:",decoder_output[1].shape)

    """

    np.random.seed(60)
    image = np.random.randn(4,1,32,128).astype("float32")
    print("inputdata:", np.sum(image))
    image = paddle.to_tensor(image)
    print(image)
    print("inputdata:", np.sum(image.numpy()))
    text_length = paddle.to_tensor([3,2,2,4])
    text_input = paddle.to_tensor([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]])
    transformer = Transformer()
    params = paddle.load('pretrain_transformer_stroke.pdparams')
    state_dict = transformer.state_dict()
    new_state_dict = {}
    for k1 in state_dict.keys():
        if k1 == "block1.1._weight":
            k2 = "block1.1.weight"
        else:
            k2 = "module."+k1
        if "mul_layernorm" in k2 and "weight" in k2:
            k2 = k2.replace("weight","a_2")
        if "mul_layernorm" in k2 and "bias" in k2:
            k2 = k2.replace("bias", "b_2")
        if "mask_multihead.linears.0.bias" in k2:
            print("--------------------")
            print("linear weight:", np.sum(params[k2].numpy()))
        if k2 not in params.keys():
            #pass
            print("The pretrained params {} not in model".format(k2))
            #print(k2)
        else:
            if list(state_dict[k1].shape) == list(params[k2].shape):
                new_state_dict[k1] = params[k2]
            else:
                print(
                    "The shape of model params {} {} not matched with loaded params {} {} !".
                    format(k1, state_dict[k1].shape, k1, params[k2].shape))
    transformer.set_state_dict(new_state_dict)
    output = transformer(image, text_length, text_input)
    print("output:", np.sum(output[0].numpy()))
    print('build success!')
    # print(output['word_result'].shape)

