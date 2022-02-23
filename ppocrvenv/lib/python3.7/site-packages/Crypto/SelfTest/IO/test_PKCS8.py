#
#  SelfTest/IO/test_PKCS8.py: Self-test for the PKCS8 module
#
# ===================================================================
#
# Copyright (c) 2014, Legrandin <helderijs@gmail.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ===================================================================

"""Self-tests for Crypto.IO.PKCS8 module"""

import unittest
from binascii import unhexlify

from Crypto.Util.py3compat import *
from Crypto.IO import PKCS8

oid_key = '1.2.840.113549.1.1.1'

# Original RSA key (in DER format)
# hexdump -v -e '32/1 "%02x" "\n"' key.der
clear_key="""
308201ab020100025a00b94a7f7075ab9e79e8196f47be707781e80dd965cf16
0c951a870b71783b6aaabbd550c0e65e5a3dfe15b8620009f6d7e5efec42a3f0
6fe20faeebb0c356e79cdec6db4dd427e82d8ae4a5b90996227b8ba54ccfc4d2
5c08050203010001025a00afa09c70d528299b7552fe766b5d20f9a221d66938
c3b68371d48515359863ff96f0978d700e08cd6fd3d8a3f97066fc2e0d5f78eb
3a50b8e17ba297b24d1b8e9cdfd18d608668198d724ad15863ef0329195dee89
3f039395022d0ebe0518df702a8b25954301ec60a97efdcec8eaa4f2e76ca7e8
8dfbc3f7e0bb83f9a0e8dc47c0f8c746e9df6b022d0c9195de13f09b7be1fdd7
1f56ae7d973e08bd9fd2c3dfd8936bb05be9cc67bd32d663c7f00d70932a0be3
c24f022d0ac334eb6cabf1933633db007b763227b0d9971a9ea36aca8b669ec9
4fcf16352f6b3dcae28e4bd6137db4ddd3022d0400a09f15ee7b351a2481cb03
09920905c236d09c87afd3022f3afc2a19e3b746672b635238956ee7e6dd62d5
022d0cd88ed14fcfbda5bbf0257f700147137bbab9c797af7df866704b889aa3
7e2e93df3ff1a0fd3490111dcdbc4c
"""

# Same key as above, wrapped in PKCS#8 but w/o password
#
# openssl pkcs8 -topk8 -inform DER -nocrypt -in key.der -outform DER -out keyp8.der
# hexdump -v -e '32/1 "%02x" "\n"' keyp8.der
wrapped_clear_key="""
308201c5020100300d06092a864886f70d0101010500048201af308201ab0201
00025a00b94a7f7075ab9e79e8196f47be707781e80dd965cf160c951a870b71
783b6aaabbd550c0e65e5a3dfe15b8620009f6d7e5efec42a3f06fe20faeebb0
c356e79cdec6db4dd427e82d8ae4a5b90996227b8ba54ccfc4d25c0805020301
0001025a00afa09c70d528299b7552fe766b5d20f9a221d66938c3b68371d485
15359863ff96f0978d700e08cd6fd3d8a3f97066fc2e0d5f78eb3a50b8e17ba2
97b24d1b8e9cdfd18d608668198d724ad15863ef0329195dee893f039395022d
0ebe0518df702a8b25954301ec60a97efdcec8eaa4f2e76ca7e88dfbc3f7e0bb
83f9a0e8dc47c0f8c746e9df6b022d0c9195de13f09b7be1fdd71f56ae7d973e
08bd9fd2c3dfd8936bb05be9cc67bd32d663c7f00d70932a0be3c24f022d0ac3
34eb6cabf1933633db007b763227b0d9971a9ea36aca8b669ec94fcf16352f6b
3dcae28e4bd6137db4ddd3022d0400a09f15ee7b351a2481cb0309920905c236
d09c87afd3022f3afc2a19e3b746672b635238956ee7e6dd62d5022d0cd88ed1
4fcfbda5bbf0257f700147137bbab9c797af7df866704b889aa37e2e93df3ff1
a0fd3490111dcdbc4c
"""

###
#
# The key above will now be encrypted with different algorithms.
# The password is always 'TestTest'.
#
# Each item in the wrapped_enc_keys list contains:
#  * wrap algorithm
#  * iteration count
#  * Salt
#  * IV
#  * Expected result
###
wrapped_enc_keys = []

#
# openssl pkcs8 -topk8 -passin pass:TestTest -inform DER -in key.der -outform DER -out keyenc.der -v2 des3
# hexdump -v -e '32/1 "%02x" "\n"' keyenc.der
#
wrapped_enc_keys.append((
'PBKDF2WithHMAC-SHA1AndDES-EDE3-CBC',
2048,
"47EA7227D8B22E2F", # IV
"E3F7A838AB911A4D", # Salt
"""
30820216304006092a864886f70d01050d3033301b06092a864886f70d01050c
300e0408e3f7a838ab911a4d02020800301406082a864886f70d0307040847ea
7227d8b22e2f048201d0ea388b374d2d0e4ceb7a5139f850fdff274884a6e6c0
64326e09d00dbba9018834edb5a51a6ae3d1806e6e91eebf33788ce71fee0637
a2ebf58859dd32afc644110c390274a6128b50c39b8d907823810ec471bada86
6f5b75d8ea04ad310fad2e73621696db8e426cd511ee93ec1714a1a7db45e036
4bf20d178d1f16bbb250b32c2d200093169d588de65f7d99aad9ddd0104b44f1
326962e1520dfac3c2a800e8a14f678dff2b3d0bb23f69da635bf2a643ac934e
219a447d2f4460b67149e860e54f365da130763deefa649c72b0dcd48966a2d3
4a477444782e3e66df5a582b07bbb19778a79bd355074ce331f4a82eb966b0c4
52a09eab6116f2722064d314ae433b3d6e81d2436e93fdf446112663cde93b87
9c8be44beb45f18e2c78fee9b016033f01ecda51b9b142091fa69f65ab784d2c
5ad8d34be6f7f1464adfc1e0ef3f7848f40d3bdea4412758f2fcb655c93d8f4d
f6fa48fc5aa4b75dd1c017ab79ac9d737233a6d668f5364ccf47786debd37334
9c10c9e6efbe78430a61f71c89948aa32cdc3cc7338cf994147819ce7ab23450
c8f7d9b94c3bb377d17a3fa204b601526317824b142ff6bc843fa7815ece89c0
839573f234dac8d80cc571a045353d61db904a4398d8ef3df5ac
"""
))

#
# openssl pkcs8 -topk8 -passin pass:TestTest -inform DER -in key.der -outform DER -out keyenc.der
# hexdump -v -e '32/1 "%02x" "\n"' keyenc.der
#
wrapped_enc_keys.append((
'skip encryption',                 # pbeWithMD5AndDES-CBC, only decoding is supported
-1,
"",
"",
"""
308201f1301b06092a864886f70d010503300e0408f9b990c89af1d41b020208
00048201d0c6267fe8592903891933d559e71a7ca68b2e39150f19daca0f7921
52f97e249d72f670d5140e9150433310ed7c7ee51927693fd39884cb9551cea5
a7b746f7edf199f8787d4787a35dad930d7db057b2118851211b645ac8b90fa6
b0e7d49ac8567cbd5fff226e87aa9129a0f52c45e9307752e8575c3b0ff756b7
31fda6942d15ecb6b27ea19370ccc79773f47891e80d22b440d81259c4c28eac
e0ca839524116bcf52d8c566e49a95ddb0e5493437279a770a39fd333f3fca91
55884fad0ba5aaf273121f893059d37dd417da7dcfd0d6fa7494968f13b2cc95
65633f2c891340193e5ec00e4ee0b0e90b3b93da362a4906360845771ade1754
9df79140be5993f3424c012598eadd3e7c7c0b4db2c72cf103d7943a5cf61420
93370b9702386c3dd4eb0a47f34b579624a46a108b2d13921fa1b367495fe345
6aa128aa70f8ca80ae13eb301e96c380724ce67c54380bbea2316c1faf4d058e
b4ca2e23442047606b9bc4b3bf65b432cb271bea4eb35dd3eb360d3be8612a87
a50e96a2264490aeabdc07c6e78e5dbf4fe3388726d0e2a228346bf3c2907d68
2a6276b22ae883fb30fa611f4e4193e7a08480fcd7db48308bacbd72bf4807aa
11fd394859f97d22982f7fe890b2e2a0f7e7ffb693
"""
))

#
# openssl pkcs8 -topk8 -passin pass:TestTest -inform DER -in key.der
#   -outform DER -out keyenc.der -v1 PBE-SHA1-RC2-64
# hexdump -v -e '32/1 "%02x" "\n"' keyenc.der
#
wrapped_enc_keys.append((
'skip encryption',                 # pbeWithSHA1AndRC2-CBC, only decoding is supported
-1,
"",
"",
"""
308201f1301b06092a864886f70d01050b300e04083ee943bdae185008020208
00048201d0e4614d9371d3ff10ceabc2f6a7a13a0f449f9a714144e46518ea55
e3e6f0cde24031d01ef1f37ec40081449ef01914faf45983dde0d2bc496712de
8dd15a5527dff4721d9016c13f34fb93e3ce68577e30146266d71b539f854e56
753a192cf126ed4812734d86f81884374f1100772f78d0646e9946407637c565
d070acab413c55952f7237437f2e48cae7fa0ff8d370de2bf446dd08049a3663
d9c813ac197468c02e2b687e7ca994cf7f03f01b6eca87dbfed94502c2094157
ea39f73fe4e591df1a68b04d19d9adab90bb9898467c1464ad20bf2b8fb9a5ff
d3ec91847d1c67fd768a4b9cfb46572eccc83806601372b6fad0243f58f623b7
1c5809dea0feb8278fe27e5560eed8448dc93f5612f546e5dd7c5f6404365eb2
5bf3396814367ae8b15c5c432b57eaed1f882c05c7f6517ee9e42b87b7b8d071
9d6125d1b52f7b2cca1f6bd5f584334bf90bce1a7d938274cafe27b68e629698
b16e27ae528db28593af9adcfccbebb3b9e1f2af5cd5531b51968389caa6c091
e7de1f1b96f0d258e54e540d961a7c0ef51fda45d6da5fddd33e9bbfd3a5f8d7
d7ab2e971de495cddbc86d38444fee9f0ac097b00adaf7802dabe0cff5b43b45
4f26b7b547016f89be52676866189911c53e2f2477"""
))

#
# openssl pkcs8 -topk8 -passin pass:TestTest -inform DER -in key.der
#   -outform DER -out keyenc.der -v1 PBE-MD5-RC2-64
# hexdump -v -e '32/1 "%02x" "\n"' keyenc.der
#
wrapped_enc_keys.append((
'skip encryption',                 # pbeWithMD5AndRC2-CBC, only decoding is supported
-1,
"",
"",
"""
308201f1301b06092a864886f70d010506300e0408f5cd2fee56d9b4b8020208
00048201d086454942d6166a19d6b108465bd111e7080911f573d54b1369c676
df28600e84936bfec04f91023ff16499e2e07178c340904f12ffa6886ab66228
32bf43c2bff5a0ed14e765918cf5fc543ad49566246f7eb3fc044fa5a9c25f40
8fc8c8296b91658d3bb1067c0aba008c4fefd9e2bcdbbbd63fdc8085482bccf4
f150cec9a084259ad441a017e5d81a1034ef2484696a7a50863836d0eeda45cd
8cee8ecabfed703f8d9d4bbdf3a767d32a0ccdc38550ee2928d7fe3fa27eda5b
5c7899e75ad55d076d2c2d3c37d6da3d95236081f9671dab9a99afdb1cbc890e
332d1a91105d9a8ce08b6027aa07367bd1daec3059cb51f5d896124da16971e4
0ca4bcadb06c854bdf39f42dd24174011414e51626d198775eff3449a982df7b
ace874e77e045eb6d7c3faef0750792b29a068a6291f7275df1123fac5789c51
27ace42836d81633faf9daf38f6787fff0394ea484bbcd465b57d4dbee3cf8df
b77d1db287b3a6264c466805be5a4fe85cfbca180699859280f2dd8e2c2c10b5
7a7d2ac670c6039d41952fbb0e4f99b560ebe1d020e1b96d02403283819c00cc
529c51f0b0101555e4c58002ba3c6e3c12e3fde1aec94382792e96d9666a2b33
3dc397b22ecab67ee38a552fec29a1d4ff8719c748"""
))

#
# openssl pkcs8 -topk8 -passin pass:TestTest -inform DER -in key.der
#   -outform DER -out keyenc.der -v1 PBE-SHA1-DES
# hexdump -v -e '32/1 "%02x" "\n"' keyenc.der
#
wrapped_enc_keys.append((
'skip encryption',                 # pbeWithSHA1AndDES-CBC, only decoding is supported
-1,
"",
"",
"""
308201f1301b06092a864886f70d01050a300e04089bacc9cf1e8f734e020208
00048201d03e502f3ceafe8fd19ab2939576bfdded26d719b2441db1459688f5
9673218b41ec1f739edf1e460bd927bc28470c87b2d4fc8ea02ba17b47a63c49
c5c1bee40529dadfd3ef8b4472c730bc136678c78abfb34670ec9d7dcd17ee3f
892f93f2629e6e0f4b24ecb9f954069bf722f466dece3913bb6abbd2c471d9a5
c5eea89b14aaccda43d30b0dd0f6eb6e9850d9747aa8aa8414c383ad01c374ee
26d3552abec9ba22669cc9622ccf2921e3d0c8ecd1a70e861956de0bec6104b5
b649ac994970c83f8a9e84b14a7dff7843d4ca3dd4af87cea43b5657e15ae0b5
a940ce5047f006ab3596506600724764f23757205fe374fee04911336d655acc
03e159ec27789191d1517c4f3f9122f5242d44d25eab8f0658cafb928566ca0e
8f6589aa0c0ab13ca7a618008ae3eafd4671ee8fe0b562e70b3623b0e2a16eee
97fd388087d2e03530c9fe7db6e52eccc7c48fd701ede35e08922861a9508d12
bc8bbf24f0c6bee6e63dbcb489b603d4c4a78ce45bf2eab1d5d10456c42a65a8
3a606f4e4b9b46eb13b57f2624b651859d3d2d5192b45dbd5a2ead14ff20ca76
48f321309aa56d8c0c4a192b580821cc6c70c75e6f19d1c5414da898ec4dd39d
b0eb93d6ba387a80702dfd2db610757ba340f63230
"""
))

#
# openssl pkcs8 -topk8 -passin pass:TestTest -inform DER -in key.der
#   -outform DER -out keyenc.der -v2 aes128
# hexdump -v -e '32/1 "%02x" "\n"' keyenc.der
#
wrapped_enc_keys.append((
'PBKDF2WithHMAC-SHA1AndAES128-CBC',
2048,
"4F66EE5D3BCD531FE6EBF4B4E73016B8", # IV
"479F25156176C53A", # Salt
"""
3082021f304906092a864886f70d01050d303c301b06092a864886f70d01050c
300e0408479f25156176c53a02020800301d060960864801650304010204104f
66ee5d3bcd531fe6ebf4b4e73016b8048201d0e33cfa560423f589d097d21533
3b880a5ebac5b2ac58b4e73b0d787aee7764f034fe34ca1d1bd845c0a7c3316f
afbfb2129e03dcaf5a5031394206492828dacef1e04639bee5935e0f46114202
10bc6c37182f4889be11c5d0486c398f4be952e5740f65de9d8edeb275e2b406
e19bc29ad5ebb97fa536344fc3d84c7e755696f12b810898de4e6f069b8a81c8
0aab0d45d7d062303aaa4a10c2ce84fdb5a03114039cfe138e38bb15b2ced717
93549cdad85e730b14d9e2198b663dfdc8d04a4349eb3de59b076ad40b116d4a
25ed917c576bc7c883c95ef0f1180e28fc9981bea069594c309f1aa1b253ceab
a2f0313bb1372bcb51a745056be93d77a1f235a762a45e8856512d436b2ca0f7
dd60fbed394ba28978d2a2b984b028529d0a58d93aba46c6bbd4ac1e4013cbaa
63b00988bc5f11ccc40141c346762d2b28f64435d4be98ec17c1884985e3807e
e550db606600993efccf6de0dfc2d2d70b5336a3b018fa415d6bdd59f5777118
16806b7bc17c4c7e20ad7176ebfa5a1aa3f6bc10f04b77afd443944642ac9cca
d740e082b4a3bbb8bafdd34a0b3c5f2f3c2aceccccdccd092b78994b845bfa61
706c3b9df5165ed1dbcbf1244fe41fc9bf993f52f7658e2f87e1baaeacb0f562
9d905c
"""
))

#
# openssl pkcs8 -topk8 -passin pass:TestTest -inform DER -in key.der
#   -outform DER -out keyenc.der -v2 aes192
# hexdump -v -e '32/1 "%02x" "\n"' keyenc.der
#
wrapped_enc_keys.append((
'PBKDF2WithHMAC-SHA1AndAES192-CBC',
2048,
"5CFC2A4FF7B63201A4A8A5B021148186", # IV
"D718541C264944CE", # Salt
"""
3082021f304906092a864886f70d01050d303c301b06092a864886f70d01050c
300e0408d718541c264944ce02020800301d060960864801650304011604105c
fc2a4ff7b63201a4a8a5b021148186048201d08e74aaa21b8bcfb15b9790fe95
b0e09ddb0f189b6fb1682fdb9f122b804650ddec3c67a1df093a828b3e5fbcc6
286abbcc5354c482fd796d972e919ca8a5eba1eaa2293af1d648013ddad72106
75622264dfba55dafdda39e338f058f1bdb9846041ffff803797d3fdf3693135
8a192729ea8346a7e5e58e925a2e2e4af0818581859e8215d87370eb4194a5ff
bae900857d4c591dbc651a241865a817eaede9987c9f9ae4f95c0bf930eea88c
4d7596e535ffb7ca369988aba75027a96b9d0bc9c8b0b75f359067fd145a378b
02aaa15e9db7a23176224da48a83249005460cc6e429168657f2efa8b1af7537
d7d7042f2d683e8271b21d591090963eeb57aea6172f88da139e1614d6a7d1a2
1002d5a7a93d6d21156e2b4777f6fc069287a85a1538c46b7722ccde591ab55c
630e1ceeb1ac42d1b41f3f654e9da86b5efced43775ea68b2594e50e4005e052
0fe753c0898120c2c07265367ff157f6538a1e4080d6f9d1ca9eb51939c9574e
f2e4e1e87c1434affd5808563cddd376776dbbf790c6a40028f311a8b58dafa2
0970ed34acd6e3e89d063987893b2b9570ddb8cc032b05a723bba9444933ebf3
c624204be72f4190e0245197d0cb772bec933fd8442445f9a28bd042d5a3a1e9
9a8a07
"""
))

#
# openssl pkcs8 -topk8 -passin pass:TestTest -inform DER -in key.der
#   -outform DER -out keyenc.der -v2 aes192
# hexdump -v -e '32/1 "%02x" "\n"' keyenc.der
#
wrapped_enc_keys.append((
'PBKDF2WithHMAC-SHA1AndAES256-CBC',
2048,
"323351F94462AC563E053A056252C2C4", # IV
"02A6CD0D12E727B5", # Salt
"""
3082021f304906092a864886f70d01050d303c301b06092a864886f70d01050c
300e040802a6cd0d12e727b502020800301d060960864801650304012a041032
3351f94462ac563e053a056252c2c4048201d07f4ef1c7be21aae738a20c5632
b8bdbbb9083b6e7f68822267b1f481fd27fdafd61a90660de6e4058790e4c912
bf3f319a7c37e6eb3d956daaa143865020d554bf6215e8d7492359aaeef45d6e
d85a686ed26c0bf7c18d071d827a86f0b73e1db0c0e7f3d42201544093302a90
551ad530692468c47ac15c69500b8ca67d4a17b64d15cecc035ae50b768a36cf
07c395afa091e9e6f86f665455fbdc1b21ad79c0908b73da5de75a9b43508d5d
44dc97a870cd3cd9f01ca24452e9b11c1b4982946702cfcbfda5b2fcc0203fb5
0b52a115760bd635c94d4c95ac2c640ee9a04ffaf6ccff5a8d953dd5d88ca478
c377811c521f2191639c643d657a9e364af88bb7c14a356c2b0b4870a23c2f54
d41f8157afff731471dccc6058b15e1151bcf84b39b5e622a3a1d65859c912a5
591b85e034a1f6af664f030a6bfc8c3d20c70f32b54bcf4da9c2da83cef49cf8
e9a74f0e5d358fe50b88acdce6a9db9a7ad61536212fc5f877ebfc7957b8bda4
b1582a0f10d515a20ee06cf768db9c977aa6fbdca7540d611ff953012d009dac
e8abd059f8e8ffea637c9c7721f817aaf0bb23403e26a0ef0ff0e2037da67d41
af728481f53443551a9bff4cea023164e9622b5441a309e1f4bff98e5bf76677
8d7cd9
"""
))

def txt2bin(inputs):
    s = b('').join([b(x) for x in inputs if not (x in '\n\r\t ')])
    return unhexlify(s)

class Rng:
    def __init__(self, output):
        self.output=output
        self.idx=0
    def __call__(self, n):
        output = self.output[self.idx:self.idx+n]
        self.idx += n
        return output

class PKCS8_Decrypt(unittest.TestCase):

    def setUp(self):
        self.oid_key = oid_key
        self.clear_key = txt2bin(clear_key)
        self.wrapped_clear_key = txt2bin(wrapped_clear_key)
        self.wrapped_enc_keys = []
        for t in wrapped_enc_keys:
            self.wrapped_enc_keys.append((
                t[0],
                t[1],
                txt2bin(t[2]),
                txt2bin(t[3]),
                txt2bin(t[4])
            ))

    ### NO ENCRYTION

    def test1(self):
        """Verify unwrapping w/o encryption"""
        res1, res2, res3 = PKCS8.unwrap(self.wrapped_clear_key)
        self.assertEqual(res1, self.oid_key)
        self.assertEqual(res2, self.clear_key)

    def test2(self):
        """Verify wrapping w/o encryption"""
        wrapped = PKCS8.wrap(self.clear_key, self.oid_key)
        res1, res2, res3 = PKCS8.unwrap(wrapped)
        self.assertEqual(res1, self.oid_key)
        self.assertEqual(res2, self.clear_key)

    ## ENCRYPTION

    def test3(self):
        """Verify unwrapping with encryption"""

        for t in self.wrapped_enc_keys:
            res1, res2, res3 = PKCS8.unwrap(t[4], b("TestTest"))
            self.assertEqual(res1, self.oid_key)
            self.assertEqual(res2, self.clear_key)

    def test4(self):
        """Verify wrapping with encryption"""

        for t in self.wrapped_enc_keys:
            if t[0] == 'skip encryption':
                continue
            rng = Rng(t[2]+t[3])
            params = { 'iteration_count':t[1] }
            wrapped = PKCS8.wrap(
                    self.clear_key,
                    self.oid_key,
                    b("TestTest"),
                    protection=t[0],
                    prot_params=params,
                    key_params=None,
                    randfunc=rng)
            self.assertEqual(wrapped, t[4])

def get_tests(config={}):
    from Crypto.SelfTest.st_common import list_test_cases
    listTests = []
    listTests += list_test_cases(PKCS8_Decrypt)
    return listTests

if __name__ == '__main__':
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')

