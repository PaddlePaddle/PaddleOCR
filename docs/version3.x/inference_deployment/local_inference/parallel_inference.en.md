# Parallel Inference for Pipelines

## Specifying Multiple Inference Devices

For some pipelines, both the CLI and Python API of PaddleOCR support specifying multiple inference devices simultaneously. If multiple devices are specified, during pipeline initialization, an instance of the underlying pipeline class will be created on each device, and the received inputs will be processed using parallel inference. For example, for the document image preprocessing pipeline:

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

Both examples above use 4 GPUs (numbered 0, 1, 2, 3) to perform parallel inference on the `doc_test_rotated.jpg` image.

When specifying multiple devices, the inference interface remains consistent with that of single-device usage. Please refer to the production line usage tutorial to check whether a specific production line supports multiple inference devices.

## Example of Multi-Process Parallel Inference

Beyond PaddleOCR's built-in multi-device parallel inference capability, users can also implement parallelism by wrapping PaddleOCR pipeline API calls themselves according to their specific scenario, with a view to achieving a better speedup. Below is an example of using Python multiprocessing to perform multi-GPU, multi-instance parallel processing on files in an input directory.



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
Assuming the script is saved as `infer_mp.py`, here are some example commands for running it:
```bash
# For the exact value of the `--pipeline` parameter, please refer to the **script** import name of the pipeline
# This is for the general layout analysis V3 pipeline, corresponding to `PPStructureV3`
# Process all files in the `input_images` directory
# Use GPUs 0, 1, 2, and 3, with 1 pipeline instance per GPU, and each instance processes 1 input file at a time
python infer_mp.py \
    --pipeline PPStructureV3 \
    --input_dir input_images/ \
    --device 'gpu:0,1,2,3' \
    --output_dir output

# General layout analysis V3 pipeline
# Process all files with the `.jpg` suffix in the `input_images` directory
# Use GPUs 0 and 2, with 2 pipeline instances per GPU, and each instance processes 4 input files at a time
python infer_mp.py \
    --pipeline PPStructureV3 \
    --input_dir input_images/ \
    --device 'gpu:0,2' \
    --output_dir output \
    --instances_per_device 2 \
    --batch_size 4 \
    --input_glob_pattern '*.jpg'

```
