import sys
import time
from pathlib import Path

import pandas as pd
from git import Repo

# Adapted from the course's batch_sim.py
RAW_FILE = "data/raw/Sample - Superstore.csv"
BATCH_DIR = Path("data/batches")
PATH_OF_GIT_REPO = r"./.git"
HISTORY_END = pd.Period("2016-12", freq="M")


class BatchLoader:
    """Simulates the arrival of new data.

    Changes from the course version:
    - reads the file with latin-1 encoding (special characters in product names)
    - first batch = the history (2014-2016), then one batch per month of 2017,
      so the pipeline always has enough data to compute indicators and train the model
    - each batch is saved in its own file, so the pipeline can read all batches so far
    """

    def __init__(self, data_file, batch_dir=BATCH_DIR):
        self.data_file = data_file
        self.batch_dir = Path(batch_dir)
        self.data = None
        self.batches = None
        self.idx = 0

    def load(self):
        """Read the raw file once, plan the batches and skip those already sent."""
        if self.data is None:
            self.data = pd.read_csv(self.data_file, encoding="latin-1")
            order_date = pd.to_datetime(self.data["Order Date"], format="%m/%d/%Y")
            self.data["_month"] = order_date.dt.to_period("M")

            months = sorted(self.data["_month"].unique())
            history = [m for m in months if m <= HISTORY_END]
            new_months = [m for m in months if m > HISTORY_END]
            self.batches = [(f"history_{history[0]}_{history[-1]}", history)]
            self.batches += [(str(m), [m]) for m in new_months]

            # Resume where the last run stopped
            self.batch_dir.mkdir(parents=True, exist_ok=True)
            self.idx = len(list(self.batch_dir.glob("*.csv")))

    def generate(self):
        """Write the next batch to data/batches/<name>.csv.
        Returns the file path, or None when every batch has been sent."""
        self.load()
        if self.idx >= len(self.batches):
            return None

        name, batch_months = self.batches[self.idx]
        selected = self.data[self.data["_month"].isin(batch_months)].drop(columns="_month")
        path = self.batch_dir / f"{name}.csv"
        selected.to_csv(path, index=False, encoding="latin-1")

        self.idx += 1
        print(f"Batch written: {path} ({len(selected)} lines)")
        return path


def git_push(message):
    """Add the new batch files, commit and push them to GitHub.
    The push is the event that triggers the GitHub Actions pipeline."""
    try:
        repo = Repo(PATH_OF_GIT_REPO)
        # Explicit path: add(update=True) would skip new, untracked batch files
        repo.git.add(str(BATCH_DIR))
        repo.index.commit(message)
        repo.remote(name="origin").push()
        print(f"Pushed: {message}")
    except Exception as error:
        print(f"Some error occurred while pushing: {error}")


if __name__ == "__main__":
    # Number of batches to send, e.g. "python batch_simulator.py 3" (default: 1)
    n_batches = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    loader = BatchLoader(RAW_FILE)
    for _ in range(n_batches):
        path = loader.generate()
        if path is None:
            print("All batches have been sent.")
            break
        git_push(f"new batch: {path.stem}")
        time.sleep(5)