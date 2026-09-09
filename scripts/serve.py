"""Production entrypoint: `python scripts/serve.py` (used by the Docker image).

Requires the repo root on `PYTHONPATH` (the Dockerfile sets it; `make serve` gets it
from the editable install).
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, workers=1)
