import argparse
import numpy as np

# 引用 paddle inference 推理库
import paddle.inference as paddle_infer

def main():
    args = parse_args()

    # 创建 config
    config = paddle_infer.Config(args.model_file, args.params_file)

    # 根据 config 创建 predictor
    predictor = paddle_infer.create_predictor(config)

    # 获取输入的名称
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    # 设置输入
    fake_input = np.random.randn(args.batch_size, 3, 318, 318).astype("float32")
    input_handle.reshape([args.batch_size, 3, 318, 318])
    input_handle.copy_from_cpu(fake_input)

    # 运行predictor
    predictor.run()

    # 获取输出
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu() # numpy.ndarray类型
    print("Output data size is {}".format(output_data.size))
    print("Output data shape is {}".format(output_data.shape))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, help="model filename")
    parser.add_argument("--params_file", type=str, help="parameter filename")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    return parser.parse_args()

if __name__ == "__main__":
    main()
