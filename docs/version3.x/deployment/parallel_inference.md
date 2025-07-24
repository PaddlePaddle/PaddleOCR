# 产线并行推理

## 指定多个推理设备

对于部分产线的 CLI 和 Python API，PaddleOCR 支持同时指定多个推理设备。如果指定了多个设备，产线初始化时将在每个设备上创建一个底层产线类对象的实例，并对接收到的输入进行并行推理。例如，对于文档图像预处理产线：

```bash
paddleocr doc_preprocessor \
  --input input_images/ \
  --device 'gpu:0,1,2,3' \
  --use_doc_orientation_classify True \
  --use_doc_unwarping True
  --save_path ./output \

```

```python
from paddleocr import DocPreprocessor


pipeline = DocPreprocessor(device="gpu:0,1,2,3") 
output = pipeline.predict(    
    input="input_images/",
    use_doc_orientation_classify=True,
    use_doc_unwarping=True)
    
```

以上两个例子均使用 4 块 GPU（编号为 0、1、2、3）对 `doc_test_rotated.jpg` 图片进行并行推理。

指定多个设备时，推理接口仍然与指定单设备时保持一致。请查看产线使用教程以了解某一产线是否支持指定多个推理设备。

## 多进程并行推理示例

除了使用 PaddleOCR 内置的多设备并行推理功能外，用户也可以结合实际场景，通过封装 PaddleOCR 产线 API 调用来实现并行处理，从而获得更优的加速效果。如下是使用 Python 多进程实现多卡、多实例并行处理输入目录中的文件的示例代码：

```python
import argparse
import sys
from multiprocessing import Manager, Process
from pathlib import Path
from queue import Empty

import paddleocr


def load_pipeline(class_name: str, device: str):
    if not hasattr(paddleocr, class_name):
        raise ValueError(f"Class {class_name} not found in paddleocr module.")
    cls = getattr(paddleocr, class_name)
    return cls(device=device)


def worker(pipeline_class_path, device, task_queue, batch_size, output_dir):
    pipeline = load_pipeline(pipeline_class_path, device)

    should_end = False
    batch = []

    while not should_end:
        try:
            input_path = task_queue.get_nowait()
        except Empty:
            should_end = True
        else:
            batch.append(input_path)

        if batch and (len(batch) == batch_size or should_end):
            try:
                for result in pipeline.predict(batch):
                    input_path = Path(result["input_path"])
                    if result.get("page_index") is not None:
                        output_path = f"{input_path.stem}_{result['page_index']}.json"
                    else:
                        output_path = f"{input_path.stem}.json"
                    output_path = str(Path(output_dir, output_path))
                    result.save_to_json(output_path)
                    print(f"Processed {repr(str(input_path))}")
            except Exception as e:
                print(
                    f"Error processing {batch} on {repr(device)}: {e}",
                    file=sys.stderr
                )
            batch.clear()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        help="PaddleOCR pipeline, e.g. 'DocPreprocessor'.",
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Input directory."
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Specifies the devices for performing parallel inference.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory."
    )
    parser.add_argument(
        "--instances_per_device",
        type=int,
        default=1,
        help="Number of pipeline instances per device.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Inference batch size for each pipeline instance.",
    )
    parser.add_argument(
        "--input_glob_pattern",
        type=str,
        default="*",
        help="Pattern to find the input files.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"The input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    if not input_dir.is_dir():
        print(f"{repr(str(input_dir))} is not a directory.", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        print(f"{repr(str(output_dir))} is not a directory.", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    from paddlex.utils.device import constr_device, parse_device

    device_type, device_ids = parse_device(args.device)
    if device_ids is None or len(device_ids) == 1:
        print(
            "Please specify at least two devices for performing parallel inference.",
            file=sys.stderr,
        )
        return 2

    if args.batch_size <= 0:
        print("Batch size must be greater than 0.", file=sys.stderr)
        return 2

    with Manager() as manager:
        task_queue = manager.Queue()
        for img_path in input_dir.glob(args.input_glob_pattern):
            task_queue.put(str(img_path))

        processes = []
        for device_id in device_ids:
            for _ in range(args.instances_per_device):
                device = constr_device(device_type, [device_id])
                p = Process(
                    target=worker,
                    args=(
                        args.pipeline,
                        device,
                        task_queue,
                        args.batch_size,
                        str(output_dir),
                    ),
                )
                p.start()
                processes.append(p)

        for p in processes:
            p.join()

    print("All done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

```

假设将上述脚本存储为 `infer_mp.py`，以下是一些调用示例：

```bash
# 确定 `--pipeline` 参数需查看其产线 **脚本方式** 导入类名称
# 此处为通用版面解析 v3 产线，对应PPStructureV3
# 处理 `input_images` 目录中所有文件
# 使用 GPU 0、1、2、3，每块 GPU 上 1 个产线实例，每个实例一次处理 1 个输入文件
python infer_mp.py \
    --pipeline PPStructureV3 \
    --input_dir input_images/ \
    --device 'gpu:0,1,2,3' \
    --output_dir output

# 通用版面解析 v3 产线
# 处理 `input_images` 目录中所有后缀为 `.jpg` 的文件
# 使用 GPU 0、2，每块 GPU 上 2 个产线实例，每个实例一次处理 4 个输入文件
python infer_mp.py \
    --pipeline PPStructureV3 \
    --input_dir input_images/ \
    --device 'gpu:0,2' \
    --output_dir output \
    --instances_per_device 2 \
    --batch_size 4 \
    --input_glob_pattern '*.jpg'
```
