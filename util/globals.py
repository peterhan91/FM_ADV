from pathlib import Path

data = {
  # Result files
  "RESULTS_DIR": "results",
  # Data files
  "DATA_DIR": "data",
  "STATS_DIR": "data/stats",
  # Hyperparameters
  "HPARAMS_DIR": "hparams",
  # Remote URLs
  "REMOTE_ROOT_URL": "https://rome.baulab.info"
}

(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR,) = (
    Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]
