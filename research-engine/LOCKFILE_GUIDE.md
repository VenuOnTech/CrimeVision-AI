# Reproducible install guide

`requirements.txt` contains bounds, not pins. Bounds prevent major-version
drift (the `transformers<5.0.0` and `huggingface_hub<2.0.0` constraints
exist because of a real incident). Bounds do not pin patch versions, so two
developers can get different `torch 2.4.0` vs `torch 2.4.1`.

## Producing a lockfile

Install pip-tools once:

```powershell
python -m pip install pip-tools