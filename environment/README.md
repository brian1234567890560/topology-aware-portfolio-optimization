# Environment

Python is the primary language because it supports market data, persistent homology, distribution distances, clustering, constrained optimization, visualization, and Jupyter in one research environment.

## Recommended Python version

Use Python 3.11 or 3.12.

## Local installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment/requirements.txt
python -m jupyter lab
```

Windows PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Google Colab

The notebook includes:

```python
%pip install -q yfinance ripser
```

Colab already provides most of the remaining scientific Python stack. Restart the runtime if package installation requests it.

## Reproducibility

- Keep the notebook seed fixed at 42 for baseline runs.
- Record the price-download date.
- Export all result CSVs.
- Record package versions with:

```bash
python -m pip freeze > portfolio_outputs/installed-packages.txt
```

- Never tune parameters using the final holding-period results.

