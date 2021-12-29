<a name="0"></a>
# Knowledge Distillation

- [Knowledge Distillation](#0)
  + [1. Introduction](#1)
    + [1.1 Introduction to Knowledge Distillation](#11)
    + [1.2 Introduction to PaddleOCR Knowledge Distillation](#12)
  + [2. Configuration File Analysis](#2)
    + [2.1 Recognition Model Configuration File Analysis](#21)
      - [2.1.1 Model Structure](#211)
      - [2.1.2 Loss Function ](#212)
      - [2.1.3 Post-processing](#213)
      - [2.1.4 Metric Calculation](#214)
      - [2.1.5 Fine-tuning Distillation Model](#215)
    + [2.2 Detection Configuration File Analysis](#22)


<a name="1"></a>
## 1. Introduction
<a name="11"></a>
### 1.1 Introduction to Knowledge Distillation

In recent years, deep neural networks have been proved to be an extremely effective method for solving problems in the fields of computer vision and natural language processing.
By constructing a suitable neural network and training it, the performance metrics of the final network model will basically exceed the traditional algorithm.
When the amount of data is large enough, increasing the amount of parameters by constructing a reasonable network model can significantly improve the performance of the model,
but this brings about the problem of a sharp increase in the complexity of the model. Large models are more expensive to use in actual scenarios.
Deep neural networks generally have more parameter redundancy. At present, there are several main methods to compress the model and reduce the amount of its parameters.
Such as pruning, quantification, knowledge distillation, etc., where knowledge distillation refers to the use of teacher models to guide student models to learn specific tasks,
to ensure that the small model obtains a relatively large performance improvement under the condition of unchanged parameters.
In addition, in the knowledge distillation task, a mutual learning model training method was also derived.
The paper [Deep Mutual Learning](https://arxiv.org/abs/1706.00384) pointed out that using two identical models to supervise each other during the training process can achieve better results than a single model training.

<a name="12"></a>
### 1.2 Introduction to PaddleOCR Knowledge Distillation

Whether it is a large model distilling a small model, or a small model learning from each other and updating parameters,
they are essentially the output between different models or mutual supervision between feature maps.
The only difference is (1) whether the model requires fixed parameters. (2) Whether the model needs to be loaded with a pre-trained model.
For the case where a large model distills a small model, the large model generally needs to load the pre-trained model and fix the parameters.
For the situation where small models distill each other, the small models generally do not load the pre-trained model, and the parameters are also in a learnable state.

In the task of knowledge distillation, it is not only the distillation between two models, but also the situation where multiple models learn from each other.
Therefore, in the knowledge distillation code framework, it is also necessary to support this type of distillation method.

The algorithm of knowledge distillation is integrated in PaddleOCR. Specifically, it has the following main features:
- It supports mutual learning of any network, and does not require the sub-network structure to be completely consistent or to have a pre-trained model. At the same time, there is no limit to the number of sub-networks, just add it in the configuration file.
- Support arbitrarily configuring the loss function through the configuration file, not only can use a certain loss, but also a combination of multiple losses.
- Support all model-related environments such as knowledge distillation training, prediction, evaluation, and export, which is convenient for use and deployment.

Through knowledge distillation, in the common Chinese and English text recognition task, without adding any time-consuming prediction,
the accuracy of the model can be improved by more than 3%. Combining the learning rate adjustment strategy and the model structure fine-tuning strategy,
the final improvement is more than 5%.

<a name="2"></a>
## 2. Configuration File Analysis

In the process of knowledge distillation training, there is no change in data preprocessing, optimizer, learning rate, and some global attributes.
The configuration files of the model structure, loss function, post-processing, metric calculation and other modules need to be fine-tuned.

The following takes the knowledge distillation configuration file for recognition and detection as an example to analyze the training and configuration of knowledge distillation.

<a name="21"></a>
### 2.1 Recognition Model Configuration File Analysis

The configuration file is in [ch_PP-OCRv2_rec_distillation.yml](../../configs/rec/ch_PP-OCRv2/ch_PP-OCRv2_rec_distillation.yml).

<a name="211"></a>
#### 2.1.1 Model Structure

In the knowledge distillation task, the model structure configuration is as follows.
```yaml
Architecture:
  model_type: &model_type "rec"    # Model category, recognition, detection, etc.
  name: DistillationModel          # Structure name, in the distillation task, it is DistillationModel
  algorithm: Distillation          # Algorithm name
  Models:                          # Model, including the configuration information of the subnet
    Teacher:                       # The name of the subnet, it must include at least the `pretrained` and `freeze_params` parameters, and the other parameters are the construction parameters of the subnet
      pretrained:                  # Does this sub-network need to load pre-training weights
      freeze_params: false         # Do you need fixed parameters
      return_all_feats: true       # Do you need to return all features, if it is False, only the final output is returned
      model_type: *model_type      # Model category
      algorithm: CRNN              # The algorithm name of the sub-network. The remaining parameters of the sub-network are consistent with the general model training configuration
      Transform:
      Backbone:
        name: MobileNetV1Enhance
        scale: 0.5
      Neck:
        name: SequenceEncoder
        encoder_type: rnn
        hidden_size: 64
      Head:
        name: CTCHead
        mid_channels: 96
        fc_decay: 0.00002
    Student:                       # Another sub-network, here is a distillation example of DML, the two sub-networks have the same structure, and both need to learn parameters
      pretrained:                  # The following parameters are the same as above
      freeze_params: false
      return_all_feats: true
      model_type: *model_type
      algorithm: CRNN
      Transform:
      Backbone:
        name: MobileNetV1Enhance
        scale: 0.5
      Neck:
        name: SequenceEncoder
        encoder_type: rnn
        hidden_size: 64
      Head:
        name: CTCHead
        mid_channels: 96
        fc_decay: 0.00002
```

If you want to add more sub-networks for training, you can also add the corresponding fields in the configuration file according to the way of adding `Student` and `Teacher`.
For example, if you want 3 models to supervise each other and train together, then `Architecture` can be written in the following format.

```yaml
Architecture:
  model_type: &model_type "rec"
  name: DistillationModel
  algorithm: Distillation
  Models:
    Teacher:
      pretrained:
      freeze_params: false
      return_all_feats: true
      model_type: *model_type
      algorithm: CRNN
      Transform:
      Backbone:
        name: MobileNetV1Enhance
        scale: 0.5
      Neck:
        name: SequenceEncoder
        encoder_type: rnn
        hidden_size: 64
      Head:
        name: CTCHead
        mid_channels: 96
        fc_decay: 0.00002
    Student:
      pretrained:
      freeze_params: false
      return_all_feats: true
      model_type: *model_type
      algorithm: CRNN
      Transform:
      Backbone:
        name: MobileNetV1Enhance
        scale: 0.5
      Neck:
        name: SequenceEncoder
        encoder_type: rnn
        hidden_size: 64
      Head:
        name: CTCHead
        mid_channels: 96
        fc_decay: 0.00002
    Student2:                       # The new sub-network introduced in the knowledge distillation task, the configuration is the same as above
      pretrained:
      freeze_params: false
      return_all_feats: true
      model_type: *model_type
      algorithm: CRNN
      Transform:
      Backbone:
        name: MobileNetV1Enhance
        scale: 0.5
      Neck:
        name: SequenceEncoder
        encoder_type: rnn
        hidden_size: 64
      Head:
        name: CTCHead
        mid_channels: 96
        fc_decay: 0.00002
```

When the model is finally trained, it contains 3 sub-networks: `Teacher`, `Student`, `Student2`.

The specific implementation code of the `DistillationModel` class can refer to [distillation_model.py](../../ppocr/modeling/architectures/distillation_model.py).
The final model output is a dictionary, the key is the name of all the sub-networks, for example, here are `Student` and `Teacher`, and the value is the output of the corresponding sub-network,
which can be `Tensor` (only the last layer of the network is returned) and `dict` (also returns the characteristic information in the middle).
In the recognition task, in order to add more loss functions and ensure the scalability of the distillation method, the output of each sub-network is saved as a `dict`, which contains the sub-module output.
Take the recognition model as an example. The output result of each sub-network is `dict`, the key contains `backbone_out`, `neck_out`, `head_out`, and `value` is the tensor of the corresponding module. Finally, for the above configuration file, `DistillationModel` The output format is as follows.

```json
{
  "Teacher": {
    "backbone_out": tensor,
    "neck_out": tensor,
    "head_out": tensor,
  },
  "Student": {
    "backbone_out": tensor,
    "neck_out": tensor,
    "head_out": tensor,
  }
}
```

<a name="212"></a>
#### 2.1.2 Loss Function

In the knowledge distillation task, the loss function configuration is as follows.

```yaml
Loss:
  name: CombinedLoss                           # Loss function name
  loss_config_list:                            # List of loss function configuration files, mandatory functions for CombinedLoss
  - DistillationCTCLoss:                       # CTC loss function based on distillation, inherited from standard CTC loss
      weight: 1.0                              # The weight of the loss function. In loss_config_list, each loss function must include this field
      model_name_list: ["Student", "Teacher"]  # For the prediction results of the distillation model, extract the output of these two sub-networks and calculate the CTC loss with gt
      key: head_out                            # In the sub-network output dict, take the corresponding tensor
  - DistillationDMLLoss:                       # DML loss function, inherited from the standard DMLLoss
      weight: 1.0  
      act: "softmax"                           # Activation function, use it to process the input, can be softmax, sigmoid or None, the default is None
      model_name_pairs:                        # The subnet name pair used to calculate DML loss. If you want to calculate the DML loss of other subnets, you can continue to add it below the list
      - ["Student", "Teacher"]
      key: head_out  
  - DistillationDistanceLoss:                  # Distilled distance loss function
      weight: 1.0  
      mode: "l2"                               # Support l1, l2 or smooth_l1
      model_name_pairs:                        # Calculate the distance loss of the subnet name pair
      - ["Student", "Teacher"]
      key: backbone_out  
```

Among the above loss functions, all distillation loss functions are inherited from the standard loss function class.
The main functions are: Analyze the output of the distillation model, find the intermediate node (tensor) used to calculate the loss,
and then use the standard loss function class to calculate.

Taking the above configuration as an example, the final distillation training loss function contains the following three parts.

- The final output `head_out` of `Student` and `Teacher` calculates the CTC loss with gt (loss weight equals 1.0). Here, because both sub-networks need to update the parameters, both of them need to calculate the loss with gt.
- DML loss between `Student` and `Teacher`'s final output `head_out` (loss weight equals 1.0).
- L2 loss between `Student` and `Teacher`'s backbone network output `backbone_out` (loss weight equals 1.0).

For more specific implementation of `CombinedLoss`, please refer to: [combined_loss.py](../../ppocr/losses/combined_loss.py#L23).
For more specific implementations of distillation loss functions such as `DistillationCTCLoss`, please refer to [distillation_loss.py](../../ppocr/losses/distillation_loss.py)


<a name="213"></a>
#### 2.1.3 Post-processing

In the knowledge distillation task, the post-processing configuration is as follows.

```yaml
PostProcess:
  name: DistillationCTCLabelDecode       # CTC decoding post-processing of distillation tasks, inherited from the standard CTCLabelDecode class
  model_name: ["Student", "Teacher"]     # For the prediction results of the distillation model, extract the outputs of these two sub-networks and decode them
  key: head_out                          # Take the corresponding tensor in the subnet output dict
```

Taking the above configuration as an example, the CTC decoding output of the two sub-networks `Student` and `Teahcer` will be calculated at the same time.
Among them, `key` is the name of the subnet, and `value` is the list of subnets.

For more specific implementation of `DistillationCTCLabelDecode`, please refer to: [rec_postprocess.py](../../ppocr/postprocess/rec_postprocess.py#L128)


<a name="214"></a>
#### 2.1.4 Metric Calculation

In the knowledge distillation task, the metric calculation configuration is as follows.

```yaml
Metric:
  name: DistillationMetric         # CTC decoding post-processing of distillation tasks, inherited from the standard CTCLabelDecode class
  base_metric_name: RecMetric      # The base class of indicator calculation. For the output of the model, the indicator will be calculated based on this class
  main_indicator: acc              # The name of the indicator
  key: "Student"                   # Select the main_indicator of this subnet as the criterion for saving the best model
```

Taking the above configuration as an example, the accuracy metric of the `Student` subnet will be used as the judgment metric for saving the best model.
At the same time, the accuracy metric of all subnets will be printed out in the log.

For more specific implementation of `DistillationMetric`, please refer to: [distillation_metric.py](../../ppocr/metrics/distillation_metric.py#L24).


<a name="215"></a>
#### 2.1.5 Fine-tuning Distillation Model

There are two ways to fine-tune the distillation recognition task.

1. Fine-tuning based on knowledge distillation: this situation is relatively simple, download the pre-trained model. Then configure the pre-training model path and your own data path in [ch_PP-OCRv2_rec_distillation.yml](../../configs/rec/ch_PP-OCRv2/ch_PP-OCRv2_rec_distillation.yml) to perform fine-tuning training of the model.
2. Do not use knowledge distillation in fine-tuning: In this case, you need to first extract the student model parameters from the pre-training model. The specific steps are as follows.

- First download the pre-trained model and unzip it.
```shell
wget https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_train.tar
tar -xf ch_PP-OCRv2_rec_train.tar
```

- Then use python to extract the student model parameters

```python
import paddle
# Load the pre-trained model
all_params = paddle.load("ch_PP-OCRv2_rec_train/best_accuracy.pdparams")
# View the keys of the weight parameter
print(all_params.keys())
# Weight extraction of student model
s_params = {key[len("Student."):]: all_params[key] for key in all_params if "Student." in key}
# View the keys of the weight parameters of the student model
print(s_params.keys())
# Save weight parameters
paddle.save(s_params, "ch_PP-OCRv2_rec_train/student.pdparams")
```

After the extraction is complete, use [ch_PP-OCRv2_rec.yml](../../configs/rec/ch_PP-OCRv2/ch_PP-OCRv2_rec.yml) to modify the path of the pre-trained model (the path of the exported `student.pdparams` model) and your own data path to fine-tune the model.

<a name="22"></a>
### 2.2 Detection Configuration File Analysis

- coming soon!
