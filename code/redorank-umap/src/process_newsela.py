"""
This script generates the base data file for REdORank's training -- newsela_cranfield.csv.

NOTE: Existence of prepped Objectionable data files required for this script to work.

The following steps are applied:
  1. Individual .txt files representing the Google search results are combined into a single file --
     newsela_search_results.csv.
  2. Any rows that are missing data are removed, and the remaining queries are saved to a new file --
     newsela_search_results_clean.csv.
  3. The originating query (NewsELA article title) is injected at the top of the result and rankings adjusted.
  4. An objectionable resource from the Alexa Top Sites by Category data is injected at the bottom of each ranking.
  5. Fully constructed frame output to a file -- newsela_cranfield.csv
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm


DATA_DIR = Path(__file__).resolve().parent.parent.joinpath("datasets")


def create_initial_data_frame():
    """
    Coalesces search results text files into a single CSV.

    Returns
    -------
    None
    """
    complete_frame = pd.DataFrame()
    results_dir = Path(DATA_DIR).joinpath("ranking", "data", "newsela", "search_results")
    for f in tqdm(results_dir.iterdir(), desc="Reading search results"):
        if f.is_file():
            # Read file, make Data Frame
            with open(f, "r") as infile:
                data = json.loads(infile.readlines()[0])
                search_results = data["search_results"]
                titles = []
                links = []
                snippets = []
                for r in search_results:
                    titles.append(r[0])
                    links.append(r[1])
                    snippets.append(r[2])

                qid_extended = [data["qid"]] * len(titles)
                frame = pd.DataFrame()
                frame["qid"] = qid_extended
                frame["title"] = titles
                frame["url"] = links
                frame["snippet"] = snippets
                frame["rank"] = np.arange(1, len(titles) + 1)
                complete_frame = pd.concat([complete_frame, frame])

    output = Path(DATA_DIR).joinpath("ranking", "data", "newsela", "newsela_search_results.csv")
    complete_frame.to_csv(output, index=False)
    print(f"{complete_frame.shape[0]} rows written to {output}")


def remove_nas():
    """
    Removes rows that are missing data in any column. Produces a new CSV with fully populated rows.

    Returns
    -------
    None.
    """
    file_path = Path(DATA_DIR).joinpath("ranking", "data", "newsela", "newsela_search_results.csv")
    frame = pd.read_csv(file_path)
    frame_no_nas = frame[~frame["title"].isna()]
    save_frame = frame_no_nas.sort_values(by="qid")
    save_frame.to_csv(Path(DATA_DIR).joinpath("ranking", "data", "newsela", "newsela_search_results_clean.csv"),
                      index=False)


def build_redorank_cranfield():
    """
    Applies Cranfield Paradigm to NewsELA search results. Performs additional step of injecting an objectionable
    resource at the bottom of each ranked list. Produces CSV with fully formed ranked lists.

    Returns
    -------
    None
    """
    new_frame = pd.DataFrame()
    data_dir = Path(DATA_DIR).joinpath("ranking", "data", "newsela")
    objectionable_content = pd.read_csv(Path(DATA_DIR).joinpath("objectionable", "data", "redorank_obj_set.csv"))
    objectionable_content = objectionable_content[objectionable_content["label"] == 1]
    newsela_all = pd.read_csv(Path(data_dir).joinpath("newsela_english.csv"))
    newsela_all = newsela_all[newsela_all["grade_level"] <= 5.0]
    newsela_used = pd.read_csv(Path(data_dir).joinpath("newsela_already_used.csv"), index_col=0)
    used_queries = list(newsela_used["query"])
    queries = newsela_all[~newsela_all["title"].isin(used_queries)]

    search_results = pd.read_csv(Path(data_dir).joinpath("newsela_search_results_clean.csv"))

    unique_queries_in_results = search_results["qid"].unique()

    for qid in tqdm(unique_queries_in_results, desc="Adding objectionable resources to the ranked lists"):
        ranked_list = search_results[search_results["qid"] == qid]

        # Check for the original within the new ranked list, and adjust ranks as needed
        original_article = queries[queries.index.isin([qid])]
        if original_article.shape[0] > 0:
            oa_title = original_article["title"].tolist()[0]
            ranked_titles = ranked_list["title"].tolist()
            has_original_article = any([oa_title == t for t in ranked_titles])
            if has_original_article:
                # Find it, change it's rank to 1 and increment everyone else's
                oa_index = ranked_list[ranked_list["title"] == oa_title].index.tolist()
                oa_from_ranked = ranked_list[ranked_list.index.isin(oa_index)]
                if oa_from_ranked["rank"].values[0] == 1:
                    continue
                else:
                    ranked_list[ranked_list.index == oa_index[0]]["rank"] = 1
                    ranked_list[~ranked_list.index.isin(oa_index)]["rank"] += 1
            else:
                # Drop the last item
                last_item = ranked_list[ranked_list["rank"] == max(ranked_list["rank"].tolist())]
                ranked_list = ranked_list.drop(int(last_item.index.to_list()[0]))

                # Insert the item. Update ranks
                new_row = pd.DataFrame()
                oa_title = original_article["title"].values[0]
                new_row["title"] = original_article["title"]
                new_row["snippet"] = original_article["text"]
                new_row["url"] = f"https://newsela.com/read/{'_'.join(oa_title.split()[:5])}"
                new_row["qid"] = qid
                new_row["rank"] = 1
                ranked_list["rank"] += 1
                ranked_list = pd.concat([ranked_list, new_row])

        # Add an objectionable resource to the end of every ranked list
        non_ideal = objectionable_content.sample(1)
        non_ideal.rename(columns={"content": "snippet"}, inplace=True)
        non_ideal.drop(columns=["label"], inplace=True)
        non_ideal["title"] = non_ideal["snippet"].values[0].split(".")[0]
        non_ideal["qid"] = qid
        non_ideal["rank"] = max(ranked_list["rank"].tolist())
        last_item = ranked_list[ranked_list["rank"] == max(ranked_list["rank"].tolist())]
        ranked_list = ranked_list.drop(int(last_item.index.to_list()[0]))
        ranked_list = pd.concat([ranked_list, non_ideal])
        new_frame = pd.concat([new_frame, ranked_list])

    new_frame.to_csv(Path(data_dir).joinpath("newsela_cranfield.csv"), index=False)


if __name__ == "__main__":
    create_initial_data_frame()
    remove_nas()
    build_redorank_cranfield()
