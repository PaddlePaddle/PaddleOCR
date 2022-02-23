# Copyright (c) 2020 VisualDL Authors. All Rights Reserve.
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
# =======================================================================

from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials
import os
# Set Host, AccessKeyID and SecretAccessKey of BosClient

bos_host = os.getenv("BOS_HOST")
access_key_id = os.getenv("BOS_AK")
secret_access_key = os.getenv("BOS_SK")


# Create BceClientConfiguration
config = BceClientConfiguration(
    credentials=BceCredentials(access_key_id, secret_access_key),
    endpoint=bos_host)
