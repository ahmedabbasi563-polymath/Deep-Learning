# UAV RF LeNet (v3) — Code

This folder contains a **single, notebook-style Python script** that mirrors the flow of the original implementation shown in the PDF (`LeNet Implementation v3.pdf`).

## Files
- `lenet_implementation_v3.py` — end-to-end pipeline (grayscale + RGB) in one script
- `requirements.txt` — minimal dependencies
- `docs/LeNet Implementation v3.pdf` — reference copy of the original write-up / notebook export

## Run
1) Install dependencies:

```bash
pip install -r requirements.txt
```

2) Edit the dataset path at the top of `lenet_implementation_v3.py`:

```python
BASE_PATH = r".../URC drones dataset v1"
```

3) Run:

```bash
python lenet_implementation_v3.py
```

## Notes
- The SNR analysis assumes the test set is ordered in contiguous SNR blocks (300 samples per SNR level).
- The RGB experiment converts normalized spectrograms to **jet-colored** images and resizes to **224×224×3**.
