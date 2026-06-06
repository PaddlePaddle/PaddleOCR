# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Módulo para carregamento e salvamento de modelos e parâmetros.

Este módulo fornece funções para carregar modelos de checkpoints ou modelos
pré-treinados, bem como para salvar modelos em diferentes formatos.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import os
import pickle
import json
from packaging import version

import paddle

from ppocr.utils.logging import get_logger
from ppocr.utils.network import maybe_download_params

try:
    import encryption
    encrypted = encryption.is_encryption_needed()
except ImportError:
    print("Skipping import of the encryption module.")
    encrypted = False

__all__ = ["load_model"]


def get_FLAGS_json_format_model():
    """
    Determina o formato do arquivo do modelo de inferência.
    
    Returns:
        bool: True se o formato JSON deve ser usado, False caso contrário.
    """
    return os.environ.get("FLAGS_json_format_model", "1").lower() in (
        "1", "true", "t"
    )


FLAGS_json_format_model = get_FLAGS_json_format_model()


def _mkdir_if_not_exist(path, logger):
    """
    Cria um diretório se ele não existir.
    
    Ignora a exceção quando múltiplos processos criam o diretório simultaneamente.
    
    Args:
        path (str): Caminho do diretório a ser criado.
        logger: Logger para mensagens de aviso.
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(
                    f"Diretório {path} já existe. Ignorando."
                )
            else:
                raise OSError(f"Falha ao criar diretório {path}")


def _load_and_set_params(model, params_path, logger):
    """
    Carrega parâmetros de um arquivo e os define no modelo.
    
    Centraliza a lógica comum de carregamento, verificação de tipos
    e atribuição de parâmetros, eliminando duplicação de código.
    
    Args:
        model: Modelo Paddle para o qual carregar parâmetros.
        params_path (str): Caminho para o arquivo de parâmetros (.pdparams).
        logger: Logger para mensagens.
        
    Returns:
        bool: True se float16 foi detectado, False caso contrário.
    """
    is_float16 = False
    
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"O arquivo de parâmetros {params_path} não existe!"
        )

    params = paddle.load(params_path)
    state_dict = model.state_dict()
    new_state_dict = {}

    for key, value in state_dict.items():
        if key not in params:
            logger.warning(f"Parâmetro {key} não encontrado!")
            continue
            
        pre_value = params[key]
        
        if pre_value.dtype == paddle.float16:
            is_float16 = True
            
        if pre_value.dtype != value.dtype:
            pre_value = pre_value.astype(value.dtype)
            
        if list(value.shape) == list(pre_value.shape):
            new_state_dict[key] = pre_value
        else:
            logger.warning(
                f"Forma incompatível para {key}: "
                f"esperado {value.shape}, obtido {pre_value.shape}"
            )

    model.set_state_dict(new_state_dict)
    return is_float16


def _save_nlp_model(arch, model_prefix, best_model_path):
    """
    Salva um modelo NLP em formato específico.
    
    Args:
        arch: Configuração da arquitetura do modelo.
        model_prefix (str): Prefixo para o caminho do modelo.
        best_model_path (str): Caminho para salvar o melhor modelo.
    """
    if "Backbone" in arch and "checkpoints" in arch["Backbone"]:
        checkpoints = arch["Backbone"]["checkpoints"]
        if checkpoints:
            paddle.jit.save(checkpoints, best_model_path)


def _save_generic_model(model, model_prefix, best_model_path):
    """
    Salva um modelo genérico em formato padrão.
    
    Args:
        model: Modelo Paddle a ser salvo.
        model_prefix (str): Prefixo para o caminho do modelo.
        best_model_path (str): Caminho para salvar o melhor modelo.
    """
    paddle.save(model.state_dict(), f"{model_prefix}.pdparams")
    
    if best_model_path:
        paddle.save(model.state_dict(), best_model_path)


def load_model(config, model, optimizer=None, model_type="det"):
    """
    Carrega modelo de checkpoint ou modelo pré-treinado.
    
    Função principal que atua como despachante, chamando funções
    auxiliares apropriadas baseado no tipo de modelo.
    
    Args:
        config (dict): Configuração do modelo.
        model: Modelo Paddle a ser carregado.
        optimizer: Otimizador (opcional).
        model_type (str): Tipo de modelo ("det", "rec", "kie", etc.).
        
    Returns:
        dict: Dicionário com informações do melhor modelo.
    """
    logger = get_logger()
    global_config = config["Global"]
    checkpoints = global_config.get("checkpoints")
    pretrained_model = global_config.get("pretrained_model")
    best_model_dict = {}
    is_float16 = False
    is_nlp_model = (
        model_type == "kie" and 
        config["Architecture"]["algorithm"] not in ["SDMGR"]
    )

    if is_nlp_model:
        # Para modelo KIE com distilação, retomar treinamento não é suportado
        if config["Architecture"]["algorithm"] in ["Distillation"]:
            return best_model_dict
            
        checkpoints = config["Architecture"]["Backbone"]["checkpoints"]
        
        # Carregar métrica KIE
        if checkpoints:
            metric_path = os.path.join(checkpoints, "metric.states")
            if os.path.exists(metric_path):
                with open(metric_path, "rb") as f:
                    states_dict = pickle.load(f, encoding="latin1")
                best_model_dict = states_dict.get("best_model_dict", {})
                if "epoch" in states_dict:
                    best_model_dict["start_epoch"] = states_dict["epoch"] + 1
                    
            logger.info(f"Retomando de {checkpoints}")

            if optimizer is not None:
                if checkpoints[-1] in ["/", "\\"]:
                    checkpoints = checkpoints[:-1]
                    
                optim_path = f"{checkpoints}.pdopt"
                if os.path.exists(optim_path):
                    optim_dict = paddle.load(optim_path)
                    optimizer.set_state_dict(optim_dict)

            is_float16 = _load_and_set_params(
                model, checkpoints, logger
            )
            return best_model_dict

    # Carregar de checkpoint
    if checkpoints:
        is_float16 = _load_and_set_params(
            model, checkpoints, logger
        )
        logger.info(f"Carregado de {checkpoints}")
        
        if optimizer is not None:
            if checkpoints[-1] in ["/", "\\"]:
                checkpoints = checkpoints[:-1]
                
            optim_path = f"{checkpoints}.pdopt"
            if os.path.exists(optim_path):
                optim_dict = paddle.load(optim_path)
                optimizer.set_state_dict(optim_dict)

    # Carregar modelo pré-treinado
    elif pretrained_model:
        is_float16 = load_pretrained_params(
            model, pretrained_model, logger
        )
        logger.info(f"Carregado modelo pré-treinado de {pretrained_model}")

    return best_model_dict


def load_pretrained_params(model, pretrained_model, logger):
    """
    Carrega parâmetros pré-treinados para o modelo.
    
    Args:
        model: Modelo Paddle.
        pretrained_model (str): Caminho ou URL do modelo pré-treinado.
        logger: Logger para mensagens.
        
    Returns:
        bool: True se float16 foi detectado, False caso contrário.
    """
    if not os.path.exists(pretrained_model):
        pretrained_model = maybe_download_params(
            pretrained_model
        )

    return _load_and_set_params(model, pretrained_model, logger)


def save_model(model, optimizer, model_prefix, is_best, best_model_path, 
               config, model_type="det"):
    """
    Salva modelo em checkpoint.
    
    Função principal que atua como despachante, chamando funções
    auxiliares apropriadas baseado no tipo de modelo.
    
    Args:
        model: Modelo Paddle a ser salvo.
        optimizer: Otimizador Paddle.
        model_prefix (str): Prefixo para o caminho do modelo.
        is_best (bool): Se é o melhor modelo até agora.
        best_model_path (str): Caminho para salvar o melhor modelo.
        config (dict): Configuração do modelo.
        model_type (str): Tipo de modelo.
    """
    logger = get_logger()
    
    if model_type == "kie":
        _save_nlp_model(config["Architecture"], model_prefix, best_model_path)
    else:
        _save_generic_model(model, model_prefix, best_model_path)

    # Salvar estado do otimizador
    paddle.save(optimizer.state_dict(), f"{model_prefix}.pdopt")
    logger.info(f"Modelo salvo em {model_prefix}")


def update_train_results(best_model_dict, metrics, epoch, is_best, 
                         best_model_path, model_prefix, save_model_func):
    """
    Atualiza resultados de treinamento e salva o melhor modelo.
    
    Args:
        best_model_dict (dict): Dicionário com informações do melhor modelo.
        metrics (dict): Métricas atuais.
        epoch (int): Época atual.
        is_best (bool): Se é o melhor modelo.
        best_model_path (str): Caminho para salvar o melhor modelo.
        model_prefix (str): Prefixo para o caminho do modelo.
        save_model_func: Função para salvar o modelo.
    """
    if is_best:
        best_model_dict.update(metrics)
        best_model_dict["epoch"] = epoch
        save_model_func(best_model_path)

