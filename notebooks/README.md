# API test notebooks

One notebook per endpoint, for exercising the service locally.

| Notebook | Endpoint |
|---|---|
| `01_computer_vision.ipynb` | `POST /computer_vision/` — all three services via the `service` field |
| `02_datascience_results_bioacoustics.ipynb` | `GET /datascience_results/bioacoustics_results/` |
| `03_datascience_results_multispectral.ipynb` | `GET /datascience_results/multispectral_results/` |
| `04_datascience_results_treetracker.ipynb` | `GET /datascience_results/treetracker_results/` |
| `05_analyses_verification_summarization.ipynb` | `POST /analyses/verification_summarization/` |

## Before running

Start the service from the repo root:

```bash
python3 src/main.py pipeline pipeline_api
```

Each notebook reads `../.env` for `API_ENDPOINT_TOKEN` and the credentials its route
needs, and targets `http://localhost:8000` unless `VT_API_BASE_URL` is set.

Every notebook follows the same shape: setup, a health check, an editable request, the
raw response, then route-specific inspection and the error cases worth confirming.

## Notes

- **`01`** covers `survivability_detection`, `content_tagging` and `content_moderation`
  in one notebook, since they are one endpoint. `image_url` takes either a bare S3
  object key or an `https://` URL to the same object.
- **`02`** reads the analytics database. **`03`** and **`04`** answer from a **live S3
  scan** and pass `live_scan=true`; without it they return **501**, because their
  database sources are not implemented. A scan is bounded by a 5-minute budget and
  returns **504** if it exceeds it.
- Outputs are not committed; clear them before committing (`jupyter nbconvert
  --clear-output --inplace notebooks/*.ipynb`) so credentials and site data stay out of
  version control.
