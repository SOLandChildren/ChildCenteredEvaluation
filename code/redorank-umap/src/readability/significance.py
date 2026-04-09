"""
File to run significance tests for readability experiments
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import kruskal


def significance(frame):
    grades = np.arange(13, dtype=float)
    # Kindergarten
    for g in grades:
        gk = frame[frame["grade_numeric"] == g]
        print(f"GRADE {g} -- {gk.shape[0]} samples")
        print(kruskal(gk['fkg_er'].values, gk["spache_original_er"], gk['spache_sven_er'].values,
                      gk['spache_allen_er'].values, gk["cli_er"].values, gk["gfog_er"].values, gk["lix_er"].values,
                      gk["rix_er"].values, gk["dale_chall_er"].values, gk["smog_er"].values))
        print("=" * 150, "\n")


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent.parent.joinpath("datasets", "readability")
    web_results = pd.read_csv(Path(data_dir).joinpath("web_readability_results.csv"))
    book_results = pd.read_csv(Path(data_dir).joinpath("books_readability_results.csv"))
    print("WEB")
    print("-" * 150)
    significance(web_results)
    print("BOOKS")
    print("-" * 150)
    significance(book_results)
