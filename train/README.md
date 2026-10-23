# Fine-tune `latin_PP-OCRv5_mobile_rec` on XFUND German

| File | Purpose |
|---|---|
| `prepare_data.py` | Crops XFUND-de word boxes into recognition crops + `train_list.txt` / `eval_list.txt` (~29.6k train, ~10k val) |
| `latin_de_finetune.yml` | Fine-tune config (single GPU, lr 1e-4, 30 epochs, batch 64) |
| `run.sh` | Downloads pretrained weights, prepares data, trains, exports inference model |
| `kaggle_train.py` | Kaggle one-cell entrypoint: installs Paddle, clones the repo, runs `run.sh`, zips the output |

## Kaggle setup (manual steps)

1. **Push this branch** to GitHub (`git push -u origin kaggle-latin-de-rec-finetune`). The repo must be public, or clone with a token instead.
2. **Upload the data as a Kaggle Dataset**: zip `train_data/XFUND/de/` (needs `de.train.json`, `de.val.json`, `de.train/`, `de.val/`) and create a dataset named e.g. `xfund-de` at kaggle.com/datasets.
3. **Create a notebook** and in the right sidebar:
   - Add Data -> your `xfund-de` dataset
   - Session options -> Accelerator: **GPU P100** (or a single T4); **Internet: On** (requires phone verification)
4. Paste `train/kaggle_train.py` into a cell and run it (or **Save Version -> Save & Run All** so it runs in the background).
5. Download `latin_de_rec_model.zip` from the notebook's Output tab. It contains the checkpoints and `latin_de_rec_infer/`.

Notes:
- If the Paddle install fails (CUDA mismatch), pick the matching wheel index from paddlepaddle.org.cn (`cu118`/`cu126`).
- To resume after an interruption, set `Global.checkpoints` to `./output/latin_de_rec_finetune/latest` in the yml.
- Chars outside the Latin dict (e.g. `■`, `○`) are ignored by the encoder.

## Local / rented GPU

```bash
pip install paddlepaddle-gpu==3.0.0 -r requirements.txt
bash train/run.sh train_data/XFUND/de
```
