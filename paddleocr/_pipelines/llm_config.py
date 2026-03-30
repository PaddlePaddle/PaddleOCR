# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Helpers for building LLM chat_bot_config from third-party provider keys."""

import os


# MiniMax models and their context window sizes (tokens).
MINIMAX_MODELS = {
    "MiniMax-M2.7": 204_800,
    "MiniMax-M2.7-highspeed": 204_800,
}

MINIMAX_DEFAULT_MODEL = "MiniMax-M2.7"
MINIMAX_BASE_URL = "https://api.minimax.io/v1"


def get_minimax_chat_bot_config(api_key=None):
    """Return a ``chat_bot_config`` dict targeting MiniMax Cloud API.

    Parameters
    ----------
    api_key : str or None
        MiniMax API key.  When *None* the ``MINIMAX_API_KEY`` environment
        variable is used as a fallback.

    Returns
    -------
    dict
        A config dict ready to be passed as ``chat_bot_config`` to any
        pipeline that accepts it (PP-ChatOCRv4-doc, PP-DocTranslation, …).

    Raises
    ------
    ValueError
        If no API key is provided and the environment variable is not set.
    """
    api_key = api_key or os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise ValueError(
            "A MiniMax API key is required.  Pass it via --minimax_api_key "
            "or set the MINIMAX_API_KEY environment variable."
        )

    return {
        "module_name": "chat_bot",
        "model_name": MINIMAX_DEFAULT_MODEL,
        "base_url": MINIMAX_BASE_URL,
        "api_type": "openai",
        "api_key": api_key,
    }
