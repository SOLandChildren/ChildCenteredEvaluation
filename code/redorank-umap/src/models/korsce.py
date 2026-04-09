import math
import textstat
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from external_repos.Demo.KSAppropriateness import AppropriatenessEstimation
from external_repos.Demo.KSCurriculum import CurriculumAlignment
from external_repos.Demo.KSObjectivity import ObjectivityEstimation
from src.metrics import NDCGScorer


def penalization_score(R, G):
    if (R > G):
        if (R >= (G + 4)):
            return 0
        else:
            return ((math.cos(0.79 * R - (G -(0.21 * G)))+1)/2)
    elif (R < G):
        if (R <= (G - 6) or R <= 0):
            return 0
        else:
            return ((math.cos(0.5236 * R - (G-(0.5236 * G))) + 1) / 2)
    else:
        return 1


def run_korsce(results_frame, result_scorer):
    dataset_dir = Path(__file__).resolve().parent.parent.parent.joinpath(
        "datasets", "ranking", "data", "test_data", "test_data")
    x_test = pd.read_csv(Path(dataset_dir).joinpath("kid-friend").joinpath("bm25-with-docs.csv"))
                         # names=["rank", "qid", "edu_prob", "readability_level", "obj_prob", "doc_id"], sep=" ")
    x_test["snippet"] = x_test["snippet"].fillna('')
    qid_test = np.array(x_test["qid"].tolist())

    # Calculate the features for everything
    tqdm.pandas(desc="Calculating readability for KORSCE")
    x_test["korsce_read"] = x_test.progress_apply(
        lambda x: penalization_score(textstat.flesch_kincaid_grade(x["snippet"]), 5), axis=1)
    tqdm.pandas(desc="Calculating appropriateness for KORSCE")
    x_test["korsce_app"] = x_test.progress_apply(
        lambda x: AppropriatenessEstimation(x['snippet'], "", "").predict(), axis=1)
    tqdm.pandas(desc="Calculating curriculum alignment for KORSCE")
    x_test["korsce_curriculum"] = CurriculumAlignment(x_test['snippet']).predict_score()
    tqdm.pandas(desc="Calculating objectivity for KORSCE")
    x_test["korsce_objectivity"] = x_test.progress_apply(
        lambda x: ObjectivityEstimation(x['snippet']).predict_score(), axis=1)
    tqdm.pandas(desc="Calculating ranking score for KORSCE")
    x_test["korsce_ranking"] = x_test.progress_apply(
        lambda x: (x["korsce_read"] * 0.25) + (x["korsce_app"] * 0.29) + (x["korsce_curriculum"] * 0.31) +
                  (x["korsce_objectivity"] * 0.15), axis=1)
    results_frame = x_test.copy()
    results_frame_with_rank_prediction = pd.DataFrame()
    unique_qids = np.unique(qid_test)
    for u in unique_qids:
        sliced = pd.DataFrame(results_frame[results_frame["qid"] == u].copy())
        sliced = sliced.sort_values(by=["korsce_ranking"], ascending=False)
        sliced_size = sliced.shape[0]
        sliced["rank_pred"] = np.arange(0, sliced_size)
        results_frame_with_rank_prediction = pd.concat([results_frame_with_rank_prediction, sliced], ignore_index=True)
    results_frame_with_rank_prediction[["qid", "doc_id", "rank_pred", "korsce_ranking"]].to_csv(
        Path(dataset_dir).joinpath("kid-friend", "kid_friend_bm25_korsce_rank_predictions.txt"), index=False, sep=" ", header=False)
    # results_frame["korsce_ndcg"] = result_scorer(y_test, x_test["korsce_ranking"].values, qid_test)
    # print(results_frame["korsce_ndcg"].mean())
    return results_frame

if __name__ == "__main__":
    run_korsce(pd.DataFrame(), NDCGScorer(k=10))
