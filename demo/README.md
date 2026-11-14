# Short tech session demo

This demo provides a quick, end-to-end OCR example for a live presentation.

## Upload a scanned PDF in your browser

Start the local upload UI:

```bash
python demo/upload_ui.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765), select a scanned PDF
(up to 20 MB), and click **Run OCR**. The UI uses compact PP-OCRv6 models for
CPU-friendly inference. The uploaded file is processed locally and removed
afterward. PaddleOCR and its inference
runtime must be installed before submitting a document. In a supported Python
environment, install them with:

```bash
python -m pip install paddlepaddle
python -m pip install -e .
```

## Run it

```bash
cd /workspaces/PaddleOCR
python demo/tech_session_demo.py --image deploy/ios_demo/PaddleOCRDemo/Resources/SampleImages/general_ocr_002.jpg --lang en
```

## What it does

- Loads a sample document image from the repo.
- Runs PaddleOCR with a lightweight configuration.
- Prints recognized text in the terminal.
- Saves a visualized result and a JSON file under `demo_output/`.

## Expected flow for a short talk

1. Show the sample image.
2. Explain that PaddleOCR can read text, detect key areas, and export structured results.
3. Run the script and point out the extracted strings in the terminal.
4. Mention the saved output files as a preview of production-style integrations.
