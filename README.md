# Tiny Transformer GPT

Educational decoder-only GPT built from scratch with PyTorch. It teaches character tokenization, token/position embeddings, causal multi-head self-attention, feed-forward blocks, cross-entropy training, checkpointing, and autoregressive generation. This is a tiny learning model, not a ChatGPT replacement.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tinygpt.py train --steps 1000
python tinygpt.py generate --prompt "Once upon a time" --tokens 200
pytest -q
```

CPU works with defaults; CUDA is auto-detected. Replace `data.txt` with text you are authorized to use. Expect limited output quality because the dataset/model are intentionally small. See code comments for each part. This educational implementation is not intended for production deployment.