# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import json
import os

from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder


class FewCLUE(DatasetBuilder):
    '''
    FewCLUE: Few-shot learning for Chinese Language Understanding Evaluation
    From: https://github.com/CLUEbenchmark/FewCLUE

    bustum:
        XiaoBu Dialogue Short Text Matching

    chid:
        Chinese IDiom Dataset for Cloze Test
        
    iflytek:
        The Microsoft Research Paraphrase Corpus dataset.
    
    tnews:
        Toutiao Short Text Classificaiton for News
    
    eprstmt:
        E-commerce Product Review Dataset for Sentiment Analysis

    ocnli:
        Original Chinese Natural Language Inference

    csldcp:
        The classification data set of Chinese science and Literature Discipline
        
    cluewsc:
        WSC Winograd
    csl:
        Paper Keyword Recognition
    '''

    BUILDER_CONFIGS = {
        'bustm': {
            'url':
            "https://bj.bcebos.com/paddlenlp/datasets/FewCLUE/fewclue_bustm.tar.gz",
            'md5': '206e037a88a57a8ca1ea157fdb756b14',
            'splits': {
                'train_0': [
                    os.path.join('fewclue_bustm', 'train_0.json'),
                    '7d90d65c5305df064cbe0ea5f55be1eb'
                ],
                'train_1': [
                    os.path.join('fewclue_bustm', 'train_1.json'),
                    '5e2ae6ce0129a39f14676d0b24090927'
                ],
                'train_2': [
                    os.path.join('fewclue_bustm', 'train_2.json'),
                    '8c94f08f6f2cc93eaeb3f0cbc58aee2d'
                ],
                'train_3': [
                    os.path.join('fewclue_bustm', 'train_3.json'),
                    '6bd32b4a15959ca037f7043e06a7663d'
                ],
                'train_4': [
                    os.path.join('fewclue_bustm', 'train_4.json'),
                    '99a92cd924e1e6b4bd7c47d561fcbfee'
                ],
                'train_few_all': [
                    os.path.join('fewclue_bustm', 'train_few_all.json'),
                    '7415f826a59eea3e4b319c70f6182f21'
                ],
                'dev_0': [
                    os.path.join('fewclue_bustm', 'dev_0.json'),
                    '703c85a4595304a707f7b7caa85974f4'
                ],
                'dev_1': [
                    os.path.join('fewclue_bustm', 'dev_1.json'),
                    'b16aa8ef45c51956be768e8e2810db4e'
                ],
                'dev_2': [
                    os.path.join('fewclue_bustm', 'dev_2.json'),
                    'c5483c83c882090314e76bb7dc1e7d5a'
                ],
                'dev_3': [
                    os.path.join('fewclue_bustm', 'dev_3.json'),
                    'bfcfdf318f72ac40095a4b671c8b8ec5'
                ],
                'dev_4': [
                    os.path.join('fewclue_bustm', 'dev_4.json'),
                    'ac061fedac0c360d08090a2e19addcae'
                ],
                'dev_few_all': [
                    os.path.join('fewclue_bustm', 'dev_few_all.json'),
                    '678159abbff4a9704001190541a45000'
                ],
                'unlabeled': [
                    os.path.join('fewclue_bustm', 'unlabeled.json'),
                    '8ebf2b2178ca6e9ad3aab09b86dfaafb'
                ],
                'test': [
                    os.path.join('fewclue_bustm', 'test.json'),
                    '28363457614d6fbfdd0487c3451eb9d1'
                ],
                'test_public': [
                    os.path.join('fewclue_bustm', 'test_public.json'),
                    'b805ad47d511d819bd723b1c63a1a2dc'
                ]
            },
            'labels': None
        },
        'chid': {
            'url':
            "https://bj.bcebos.com/paddlenlp/datasets/FewCLUE/fewclue_chid.tar.gz",
            'md5': '31d209e1bda2703708f2a53da66ca6ef',
            'splits': {
                'train_0': [
                    os.path.join('fewclue_chid', 'train_0.json'),
                    '9fe1b1e9c2174c34bf2470b2b27e0d12'
                ],
                'train_1': [
                    os.path.join('fewclue_chid', 'train_1.json'),
                    '3a3971f28707250a65a3cbdeb7c40711'
                ],
                'train_2': [
                    os.path.join('fewclue_chid', 'train_2.json'),
                    'ab65bd8ca1ad1a4d464f0fd50adb5e24'
                ],
                'train_3': [
                    os.path.join('fewclue_chid', 'train_3.json'),
                    '5ac78bc3bf2dbfff754a997298abae54'
                ],
                'train_4': [
                    os.path.join('fewclue_chid', 'train_4.json'),
                    '9c3ad59e850bc2133d45d3d57353ba2c'
                ],
                'train_few_all': [
                    os.path.join('fewclue_chid', 'train_few_all.json'),
                    '5d14b6e6aa7cbc77f0ea21d9bf36e740'
                ],
                'dev_0': [
                    os.path.join('fewclue_chid', 'dev_0.json'),
                    'd50b501c0d80da404b09a3899feae907'
                ],
                'dev_1': [
                    os.path.join('fewclue_chid', 'dev_1.json'),
                    'e00c8c98dd9d79f47fd38f012c80c23b'
                ],
                'dev_2': [
                    os.path.join('fewclue_chid', 'dev_2.json'),
                    '283a68c62042f99740fc16d77d9df749'
                ],
                'dev_3': [
                    os.path.join('fewclue_chid', 'dev_3.json'),
                    '09ddb889c668368ee5842ff1f6611817'
                ],
                'dev_4': [
                    os.path.join('fewclue_chid', 'dev_4.json'),
                    'c4162fe8593fd91623c17abc7b0a0532'
                ],
                'dev_few_all': [
                    os.path.join('fewclue_chid', 'dev_few_all.json'),
                    '6e0d456dc6d103f0db677cda3b607e20'
                ],
                'unlabeled': [
                    os.path.join('fewclue_chid', 'unlabeled.json'),
                    'e4772b7600b348e9ff2245cef6a00812'
                ],
                'test': [
                    os.path.join('fewclue_chid', 'test.json'),
                    'bf46b7a643b51f64dd890e3fcae8802a'
                ],
                'test_public': [
                    os.path.join('fewclue_chid', 'test_public.json'),
                    'c8c3765c4319e370f752b601b9f2fb80'
                ]
            },
            'labels': None
        },
        'iflytek': {
            'url':
            "https://bj.bcebos.com/paddlenlp/datasets/FewCLUE/fewclue_iflytek.tar.gz",
            'md5': '6f60fd6e0ab35c934732e41b7b7489b7',
            'splits': {
                'train_0': [
                    os.path.join('fewclue_iflytek', 'train_0.json'),
                    '43e5f8ab327ae5f446fc0cfd97b6341d'
                ],
                'train_1': [
                    os.path.join('fewclue_iflytek', 'train_1.json'),
                    'b3c04b6eec6f82e53f2a913b2487974a'
                ],
                'train_2': [
                    os.path.join('fewclue_iflytek', 'train_2.json'),
                    'a4fdb0055ef1cb5543fef932a88092d0'
                ],
                'train_3': [
                    os.path.join('fewclue_iflytek', 'train_3.json'),
                    'b8626c171555afb8e25d78b32cc2cfb1'
                ],
                'train_4': [
                    os.path.join('fewclue_iflytek', 'train_4.json'),
                    '91dde0c9c939a3bc7768b105427cb3ef'
                ],
                'train_few_all': [
                    os.path.join('fewclue_iflytek', 'train_few_all.json'),
                    'db4ceaf7e6682be02f4a9e9138fcda8c'
                ],
                'dev_0': [
                    os.path.join('fewclue_iflytek', 'dev_0.json'),
                    '0703cb79c0c4fcb120c2cdeea2c56a6c'
                ],
                'dev_1': [
                    os.path.join('fewclue_iflytek', 'dev_1.json'),
                    'a4b975f7ee524e1479d2067118fe15f5'
                ],
                'dev_2': [
                    os.path.join('fewclue_iflytek', 'dev_2.json'),
                    'c0280a2675012bea323a36eb28ba2ecc'
                ],
                'dev_3': [
                    os.path.join('fewclue_iflytek', 'dev_3.json'),
                    'ffdd7073ae25e40a8fa2c95f50f71c1f'
                ],
                'dev_4': [
                    os.path.join('fewclue_iflytek', 'dev_4.json'),
                    '9e9a93fe76653ab7ee587b67061930ac'
                ],
                'dev_few_all': [
                    os.path.join('fewclue_iflytek', 'dev_few_all.json'),
                    '86ec5c85c126e8e91efc274e79c39752'
                ],
                'unlabeled': [
                    os.path.join('fewclue_iflytek', 'unlabeled.json'),
                    '431e0c787373b25f877e2c7b2fc91f91'
                ],
                'test': [
                    os.path.join('fewclue_iflytek', 'test.json'),
                    'ea764519ddb4369767d07664afde3325'
                ],
                'test_public': [
                    os.path.join('fewclue_iflytek', 'test_public.json'),
                    'b8ec7c77457baa842666f6e6620ab8fd'
                ]
            },
            'labels': None
        },
        'tnews': {
            'url':
            "https://bj.bcebos.com/paddlenlp/datasets/FewCLUE/fewclue_tnews.tar.gz",
            'md5': 'c1682c753e504fdba28328c0c9298e84',
            'splits': {
                'train_0': [
                    os.path.join('fewclue_tnews', 'train_0.json'),
                    'e540cbcbf224e9c2e8c1297abab37d1d'
                ],
                'train_1': [
                    os.path.join('fewclue_tnews', 'train_1.json'),
                    '019bb64e35371f6093451a8e7c720d02'
                ],
                'train_2': [
                    os.path.join('fewclue_tnews', 'train_2.json'),
                    '9403d45f1b65fdbea38503e842e0e915'
                ],
                'train_3': [
                    os.path.join('fewclue_tnews', 'train_3.json'),
                    '2f05be9b4f4c3b4fb468864f092005ac'
                ],
                'train_4': [
                    os.path.join('fewclue_tnews', 'train_4.json'),
                    'ced405a502292f84f305214191cbd8d0'
                ],
                'train_few_all': [
                    os.path.join('fewclue_tnews', 'train_few_all.json'),
                    '274340c49822c9cf06286bd74744cad4'
                ],
                'dev_0': [
                    os.path.join('fewclue_tnews', 'dev_0.json'),
                    'ee20628d0d544869f9cc5442658602e4'
                ],
                'dev_1': [
                    os.path.join('fewclue_tnews', 'dev_1.json'),
                    '15bd699553c8742f5d15909bf0aecddb'
                ],
                'dev_2': [
                    os.path.join('fewclue_tnews', 'dev_2.json'),
                    'f8493a1e89d9a1e915700f0a46dda861'
                ],
                'dev_3': [
                    os.path.join('fewclue_tnews', 'dev_3.json'),
                    '8948af6083f5d69ccbd1c6a9f2cc9ea6'
                ],
                'dev_4': [
                    os.path.join('fewclue_tnews', 'dev_4.json'),
                    '508790da261bfd83beffcc64fef3aa66'
                ],
                'dev_few_all': [
                    os.path.join('fewclue_tnews', 'dev_few_all.json'),
                    '9b079af311d8ccfb9938eb3f11b27ea7'
                ],
                'unlabeled': [
                    os.path.join('fewclue_tnews', 'unlabeled.json'),
                    '6ce9e45f56521fd80e32980ef73fa7b7'
                ],
                'test': [
                    os.path.join('fewclue_tnews', 'test.json'),
                    'd21791d746cd0035eaeeef9b3b9f9487'
                ],
                'test_public': [
                    os.path.join('fewclue_tnews', 'test_public.json'),
                    '5539e4a3f0abc2aa4f84da04bf02ca0d'
                ]
            },
            'labels': None
        },
        'eprstmt': {
            'url':
            "https://bj.bcebos.com/paddlenlp/datasets/FewCLUE/fewclue_eprstmt.tar.gz",
            'md5': '016091564b689fd36f52eab5e1e5407c',
            'splits': {
                'train_0': [
                    os.path.join('fewclue_eprstmt', 'train_0.json'),
                    'd027ef9d3a19b4939c6bab3013397f16'
                ],
                'train_1': [
                    os.path.join('fewclue_eprstmt', 'train_1.json'),
                    'aa70803b42143c648e127f5091c89512'
                ],
                'train_2': [
                    os.path.join('fewclue_eprstmt', 'train_2.json'),
                    'acafc32e7c241300b943fd2557c6aacf'
                ],
                'train_3': [
                    os.path.join('fewclue_eprstmt', 'train_3.json'),
                    '1cabd524e83259037f2192d978a7a32b'
                ],
                'train_4': [
                    os.path.join('fewclue_eprstmt', 'train_4.json'),
                    '8648c607f00da8f2235e744a86f44c8f'
                ],
                'train_few_all': [
                    os.path.join('fewclue_eprstmt', 'train_few_all.json'),
                    '72e4f19448bfb3b01229c3cd94d4e3e7'
                ],
                'dev_0': [
                    os.path.join('fewclue_eprstmt', 'dev_0.json'),
                    'b6aab58bc487ad6174118d8ccf87a9e1'
                ],
                'dev_1': [
                    os.path.join('fewclue_eprstmt', 'dev_1.json'),
                    '41a18a4b4d0c567c6568ff4577dbec0a'
                ],
                'dev_2': [
                    os.path.join('fewclue_eprstmt', 'dev_2.json'),
                    '618590661a58ea660cabff917cc41044'
                ],
                'dev_3': [
                    os.path.join('fewclue_eprstmt', 'dev_3.json'),
                    '18274080ad1d6612582f89065c1f19af'
                ],
                'dev_4': [
                    os.path.join('fewclue_eprstmt', 'dev_4.json'),
                    'd5d8017e3838b6184e648696fe65fbb3'
                ],
                'dev_few_all': [
                    os.path.join('fewclue_eprstmt', 'dev_few_all.json'),
                    '9cbda31b17f3adcb32ea89b020209806'
                ],
                'unlabeled': [
                    os.path.join('fewclue_eprstmt', 'unlabeled.json'),
                    'e8802dad5889d7cc8f085f7d39aeb33b'
                ],
                'test': [
                    os.path.join('fewclue_eprstmt', 'test.json'),
                    '05282edba3283a791167d0ce0343d182'
                ],
                'test_public': [
                    os.path.join('fewclue_eprstmt', 'test_public.json'),
                    '704c551bc35d7fb2e4548637b11dabec'
                ]
            },
            'labels': None
        },
        'ocnli': {
            'url':
            "https://bj.bcebos.com/paddlenlp/datasets/FewCLUE/fewclue_ocnli.tar.gz",
            'md5': 'a49a160987d67d26e217b98edeee44a9',
            'splits': {
                'train_0': [
                    os.path.join('fewclue_ocnli', 'train_0.json'),
                    '45a9a144919efde95aa53dc8b8ba9748'
                ],
                'train_1': [
                    os.path.join('fewclue_ocnli', 'train_1.json'),
                    'a63b358e1b9e3ecf833a174d65713e11'
                ],
                'train_2': [
                    os.path.join('fewclue_ocnli', 'train_2.json'),
                    '7882feb198022fe3cb6338f3652a5216'
                ],
                'train_3': [
                    os.path.join('fewclue_ocnli', 'train_3.json'),
                    '0c6321202ca1fca9843259e6b1e83f5b'
                ],
                'train_4': [
                    os.path.join('fewclue_ocnli', 'train_4.json'),
                    'f0c272e4a846b9f2483d70314a2fdff4'
                ],
                'train_few_all': [
                    os.path.join('fewclue_ocnli', 'train_few_all.json'),
                    'f6d9b9198884d3a27249b346933661b6'
                ],
                'dev_0': [
                    os.path.join('fewclue_ocnli', 'dev_0.json'),
                    '99f4dff1afabe4eb6808cc3e5bc5f422'
                ],
                'dev_1': [
                    os.path.join('fewclue_ocnli', 'dev_1.json'),
                    '4f3b1d87ebf082ef71d29e76d9aaf909'
                ],
                'dev_2': [
                    os.path.join('fewclue_ocnli', 'dev_2.json'),
                    '4c3c103f663a84f5c4fc04ee6aef98fb'
                ],
                'dev_3': [
                    os.path.join('fewclue_ocnli', 'dev_3.json'),
                    '73687b04ae00f8750981ed3f86ef0baa'
                ],
                'dev_4': [
                    os.path.join('fewclue_ocnli', 'dev_4.json'),
                    'b029f7b3f6d4681f4416fa2bc146e227'
                ],
                'dev_few_all': [
                    os.path.join('fewclue_ocnli', 'dev_few_all.json'),
                    'f0235528abf52543c0fdec7f27dd70ae'
                ],
                'unlabeled': [
                    os.path.join('fewclue_ocnli', 'unlabeled.json'),
                    '3db8319afb94780d04bfc7dff57efe81'
                ],
                'test': [
                    os.path.join('fewclue_ocnli', 'test.json'),
                    'a82e69d8372ef99537c64aacba10dd4b'
                ],
                'test_public': [
                    os.path.join('fewclue_ocnli', 'test_public.json'),
                    'ce8229a27a6948a63a3492d6acd6ee1f'
                ]
            },
            'labels': None
        },
        'csldcp': {
            'url':
            "https://bj.bcebos.com/paddlenlp/datasets/FewCLUE/fewclue_csldcp.tar.gz",
            'md5': '5ce33afe9b4b8104e028e04a97e70d5c',
            'splits': {
                'train_0': [
                    os.path.join('fewclue_csldcp', 'train_0.json'),
                    'ca5fc102bcbd5820743ef08ef415acfb'
                ],
                'train_1': [
                    os.path.join('fewclue_csldcp', 'train_1.json'),
                    'ddfeab5c1c0b7051f3d8863b5145c0b6'
                ],
                'train_2': [
                    os.path.join('fewclue_csldcp', 'train_2.json'),
                    '67fefbbabb063247108623ed9cb8bb90'
                ],
                'train_3': [
                    os.path.join('fewclue_csldcp', 'train_3.json'),
                    'eebc7bc760422dd8ff8eefd5de39995b'
                ],
                'train_4': [
                    os.path.join('fewclue_csldcp', 'train_4.json'),
                    '82ad233a803fd0e6ec4d9245299c3389'
                ],
                'train_few_all': [
                    os.path.join('fewclue_csldcp', 'train_few_all.json'),
                    '3576c8413a9c77e20360296996f1217c'
                ],
                'dev_0': [
                    os.path.join('fewclue_csldcp', 'dev_0.json'),
                    '24e6b62a23dda83ab2aa4d63b64d9306'
                ],
                'dev_1': [
                    os.path.join('fewclue_csldcp', 'dev_1.json'),
                    '73f4439696f1c447c04ad2ea873fb603'
                ],
                'dev_2': [
                    os.path.join('fewclue_csldcp', 'dev_2.json'),
                    '7f12d47d173c4beb77c4995a1409ad61'
                ],
                'dev_3': [
                    os.path.join('fewclue_csldcp', 'dev_3.json'),
                    '35936d8347dd3d727050004cb871e686'
                ],
                'dev_4': [
                    os.path.join('fewclue_csldcp', 'dev_4.json'),
                    '2fe45b969c8c33298c53c7415be9fc40'
                ],
                'dev_few_all': [
                    os.path.join('fewclue_csldcp', 'dev_few_all.json'),
                    '17078e738790997cf0fe50ebe0568b8e'
                ],
                'unlabeled': [
                    os.path.join('fewclue_csldcp', 'unlabeled.json'),
                    'e8802dad5889d7cc8f085f7d39aeb33b'
                ],
                'test': [
                    os.path.join('fewclue_csldcp', 'test.json'),
                    '8e4c1680a30da48979f684edd4d175f2'
                ],
                'test_public': [
                    os.path.join('fewclue_csldcp', 'test_public.json'),
                    '695058c4e6dc5e823be772963974c965'
                ]
            },
            'labels': None
        },
        'cluewsc': {
            'url':
            "https://bj.bcebos.com/paddlenlp/datasets/FewCLUE/fewclue_cluewsc.tar.gz",
            'md5': '328e60d2ac14aaa6ecf255a9546e538d',
            'splits': {
                'train_0': [
                    os.path.join('fewclue_cluewsc', 'train_0.json'),
                    '623085e169c6515a05cae6b52f2c5a2c'
                ],
                'train_1': [
                    os.path.join('fewclue_cluewsc', 'train_1.json'),
                    'b30acf58e613ee21cd2d6fb4833e2763'
                ],
                'train_2': [
                    os.path.join('fewclue_cluewsc', 'train_2.json'),
                    'ef0840acb8b61d22f1da7e94ecd7a309'
                ],
                'train_3': [
                    os.path.join('fewclue_cluewsc', 'train_3.json'),
                    '7e6e15afab20ae488256278fa84468b5'
                ],
                'train_4': [
                    os.path.join('fewclue_cluewsc', 'train_4.json'),
                    '5f21307270e83d3ea7d1e833db3dc514'
                ],
                'train_few_all': [
                    os.path.join('fewclue_cluewsc', 'train_few_all.json'),
                    '0f875905c77747007e6e722e27e069f9'
                ],
                'dev_0': [
                    os.path.join('fewclue_cluewsc', 'dev_0.json'),
                    'd52f7e97197af8782319be2946226b0f'
                ],
                'dev_1': [
                    os.path.join('fewclue_cluewsc', 'dev_1.json'),
                    'd602d73dd7cc4f5e421fa0fd1deccc00'
                ],
                'dev_2': [
                    os.path.join('fewclue_cluewsc', 'dev_2.json'),
                    '405bab04b2fdd00f4e23492ae24233ac'
                ],
                'dev_3': [
                    os.path.join('fewclue_cluewsc', 'dev_3.json'),
                    '6896cee55db9539687ac788430319c53'
                ],
                'dev_4': [
                    os.path.join('fewclue_cluewsc', 'dev_4.json'),
                    'a171a69d92408ce19449ddc4d629534e'
                ],
                'dev_few_all': [
                    os.path.join('fewclue_cluewsc', 'dev_few_all.json'),
                    '9d5e5066758ac6ff24534b13dd2ed1ba'
                ],
                'unlabeled': [
                    os.path.join('fewclue_cluewsc', 'unlabeled.json'),
                    'd41d8cd98f00b204e9800998ecf8427e'
                ],
                'test': [
                    os.path.join('fewclue_cluewsc', 'test.json'),
                    '0e9e8ffd8ee90ddf1f58d6dc2e02de7b'
                ],
                'test_public': [
                    os.path.join('fewclue_cluewsc', 'test_public.json'),
                    '027bc101f000b632ef45ed6d86907527'
                ]
            },
            'labels': None
        },
        'csl': {
            'url':
            "https://bj.bcebos.com/paddlenlp/datasets/FewCLUE/fewclue_csl.tar.gz",
            'md5': '434f3bad2958bba763506e9af8bf0419',
            'splits': {
                'train_0': [
                    os.path.join('fewclue_csl', 'train_0.json'),
                    'd93bf9fcef2d5839819a7c1a695d38cb'
                ],
                'train_1': [
                    os.path.join('fewclue_csl', 'train_1.json'),
                    'c5d1ce67e0c9081a160e0a0c790bc6af'
                ],
                'train_2': [
                    os.path.join('fewclue_csl', 'train_2.json'),
                    '9fe5568b97e990e68770f00ce1ecd9bf'
                ],
                'train_3': [
                    os.path.join('fewclue_csl', 'train_3.json'),
                    'e45acff15ae461bdf4001dd9f87ac413'
                ],
                'train_4': [
                    os.path.join('fewclue_csl', 'train_4.json'),
                    'db87a6229793584e3ae1cbdb173de9db'
                ],
                'train_few_all': [
                    os.path.join('fewclue_csl', 'train_few_all.json'),
                    '4b8882f1cfbdb0556b990b378ae7671e'
                ],
                'dev_0': [
                    os.path.join('fewclue_csl', 'dev_0.json'),
                    '5ef6c4cce5cd8b313bd21dd2232bbdf2'
                ],
                'dev_1': [
                    os.path.join('fewclue_csl', 'dev_1.json'),
                    'cbc3dbc4ed06bfe8bc9a9c25fdf98693'
                ],
                'dev_2': [
                    os.path.join('fewclue_csl', 'dev_2.json'),
                    '581f5db0e79beb0e8a5f43db52fc1ff3'
                ],
                'dev_3': [
                    os.path.join('fewclue_csl', 'dev_3.json'),
                    'cd3d99f6edf1ae20624b0b7aea1eeeba'
                ],
                'dev_4': [
                    os.path.join('fewclue_csl', 'dev_4.json'),
                    '765bd899bad409812e6090330fc1be13'
                ],
                'dev_few_all': [
                    os.path.join('fewclue_csl', 'dev_few_all.json'),
                    '2d4c44445a25bb61a48261cabea97e51'
                ],
                'unlabeled': [
                    os.path.join('fewclue_csl', 'unlabeled.json'),
                    '2582af170971ab780d5650e75842e40c'
                ],
                'test': [
                    os.path.join('fewclue_csl', 'test.json'),
                    'd34119b97113000988f1e03f92eb2dfe'
                ],
                'test_public': [
                    os.path.join('fewclue_csl', 'test_public.json'),
                    '45a97013acfe94c887cf85e6ff540456'
                ]
            },
            'labels': None
        },
    }

    def _get_data(self, mode, **kwargs):
        builder_config = self.BUILDER_CONFIGS[self.name]

        default_root = os.path.join(DATA_HOME, self.__class__.__name__)

        filename, data_hash = builder_config['splits'][mode]

        fullname = os.path.join(default_root, filename)

        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            get_path_from_url(builder_config['url'], default_root,
                              builder_config['md5'])
        return fullname

    def _read(self, filename, split):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line.rstrip())

    def get_labels(self):
        """
        Return labels of the FewCLUE task.
        """
        return self.BUILDER_CONFIGS[self.name]['labels']
