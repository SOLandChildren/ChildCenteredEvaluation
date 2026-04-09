import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
from pathlib import Path
import random
from tqdm import tqdm
from readability.formulas import spache_allen
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from build_letor import write_training_files
HF_ACCESS_TOKEN = os.environ.get("HF_ACCESS_TOKEN")

DATA_DIR = Path(__file__).resolve().parent.parent.joinpath("datasets")

educational_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/fineweb-edu-classifier", token=HF_ACCESS_TOKEN)
educational_classifier = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/fineweb-edu-classifier", token=HF_ACCESS_TOKEN)
obj_pipe = pipeline("text-classification", "GroNLP/mdebertav3-subjectivity-english", top_k=2)

def classify_educational(text):
    inputs = educational_tokenizer(text, return_tensors="pt", padding="longest", truncation=True)
    outputs = educational_classifier(**inputs)
    logits = outputs.logits.squeeze(-1).float().detach().numpy()
    score = logits.item()
    int_score = int(round(max(0, min(score, 5))))
    normalized_int_score = int_score / 5.0
    return normalized_int_score


def classify_objectivity(text):
    pipe_pred = obj_pipe(text)
    frame = pd.DataFrame(pipe_pred[0])
    score = frame[frame["label"] == "LABEL_1"]["score"].values[0]
    return score


def get_text(row, documents_frame):
    content = documents_frame[documents_frame["docno"] == row["docno"]]
    try:
        text = content["contents"].values[0]
    except KeyError:
        text = content["snippet"].values[0]
    return text


def feature_generation(dataframe):
    # Readability
    tqdm.pandas(desc="Calculating readability level", unit=" resources")
    dataframe["reading_level"] = dataframe["snippet"].progress_apply(spache_allen)

    # Educational
    tqdm.pandas(desc="Calculating educational level", unit=" resources")
    dataframe["edu_prob"] = dataframe["snippet"].progress_apply(classify_educational)

    # Objectivity
    tqdm.pandas(desc="Calculating objectivity level", unit=" resources")
    dataframe["obj_prob"] = dataframe["snippet"].progress_apply(classify_objectivity)

    return dataframe


def process_requik_bm25():
    requik_rank = Path(__file__).parent.parent.joinpath(
        "datasets", "ranking", "data", "requik")
    requik_parent = Path(__file__).parent.parent.joinpath(
        "datasets", "ranking", "data", "test_data", "test_data", "requik")
    requik_docs_path = Path(__file__).parent.parent.joinpath(
        "datasets", "ranking", "data", "test_data", "test_data", "requik", "corpus", "documents.jsonl")
    # requik_qrels_path = Path(__file__).parent.parent.joinpath(
    #     "datasets", "ranking", "data", "test_data", "test_data", "requik", "qrels", "qrels-relevance-binary.csv")
    ranked = pd.read_csv(Path(requik_rank).joinpath("requik_BM25.res"), names=["qid", "q0", "docno", "rank", "bm25_score", "ranker"], sep=" ")
    docs = pd.read_json(requik_docs_path, lines=True)
    docs.rename(columns={"id":"docno"}, inplace=True)
    combined = ranked.copy()
    combined["snippet"] = combined.apply(get_text, args=(docs,), axis=1)
    combined.rename(columns={"docno":"doc_id"}, inplace=True)
    combined.to_csv(Path(requik_parent).joinpath("bm25-with-docs.csv"), index=False)
    combined["query_id"] = combined["qid"]
    combined["qid"] = combined.apply(lambda x: int(x["qid"].replace("RQq", "")), axis=1)

    # Process the documents for features to be written into a svm_light file
    data = feature_generation(combined)

    features = {
        "train_data": pd.DataFrame(),
        "test_data": data,
        "val_data": pd.DataFrame(),
        "dev_train_data": pd.DataFrame(),
        "dev_test_data": pd.DataFrame(),
        "dev_val_data": pd.DataFrame()
    }

    requik_all_path = Path(Path(requik_rank).joinpath("all_features"))
    write_training_files(features, requik_all_path, "requik", all_feats=True)


def process_kid_friend_bing():
    kf_rank = Path(DATA_DIR).joinpath("ranking", "data", "kid-friend")
    kf_parent = Path(DATA_DIR).joinpath("ranking", "data", "test_data", "test_data", "kid-friend")
    kf_docs_path = Path(DATA_DIR).joinpath(
        "ranking", "data", "test_data", "test_data", "kid-friend", "corpus", "documents.jsonl")
    ranked = pd.read_csv(Path(kf_rank).joinpath("bing.txt"),
                         names=["qid", "q0", "docno", "rank", "bing_score", "ranker"], sep=" ")
    docs = pd.read_json(kf_docs_path, lines=True)
    docs.rename(columns={"id": "docno"}, inplace=True)
    combined = ranked.copy()
    combined["snippet"] = combined.apply(get_text, args=(docs,), axis=1)
    combined.rename(columns={"docno": "doc_id"}, inplace=True)
    combined.to_csv(Path(kf_parent).joinpath("bing-with-docs.csv"), index=False)

    # Process the documents for features to be written into a svm_light file
    data = feature_generation(combined)
    features = {
        "train_data": pd.DataFrame(),
        "test_data": data,
        "val_data": pd.DataFrame(),
        "dev_train_data": pd.DataFrame(),
        "dev_test_data": pd.DataFrame(),
        "dev_val_data": pd.DataFrame()
    }

    kf_all_path = Path(Path(kf_rank).joinpath("all_features"))
    write_training_files(features, kf_all_path, "kid-friend", all_feats=True)

    # rename the file to avoid overwriting conflicts
    current_path = Path(kf_all_path).joinpath(f"kid-friend_test_data.txt")
    new_path = Path(kf_all_path).joinpath(f"kid-friend_bing_test_data.txt")
    current_path.rename(new_path)


def process_kid_friend_google():
    kf_rank = Path(DATA_DIR).joinpath("ranking", "data", "kid-friend")
    kf_parent = Path(DATA_DIR).joinpath("ranking", "data", "test_data", "test_data", "kid-friend")
    kf_docs_path = Path(DATA_DIR).joinpath(
        "ranking", "data", "test_data", "test_data", "kid-friend", "corpus", "documents.jsonl")
    ranked = pd.read_csv(Path(kf_rank).joinpath("google.txt"),
                         names=["qid", "q0", "docno", "rank", "google_score", "ranker"], sep=" ")
    docs = pd.read_json(kf_docs_path, lines=True)
    docs.rename(columns={"id": "docno"}, inplace=True)
    combined = ranked.copy()
    combined["snippet"] = combined.apply(get_text, args=(docs,), axis=1)
    combined.rename(columns={"docno": "doc_id"}, inplace=True)
    combined.to_csv(Path(kf_parent).joinpath("google-with-docs.csv"), index=False)

    # Process the documents for features to be written into a svm_light file
    data = feature_generation(combined)
    features = {
        "train_data": pd.DataFrame(),
        "test_data": data,
        "val_data": pd.DataFrame(),
        "dev_train_data": pd.DataFrame(),
        "dev_test_data": pd.DataFrame(),
        "dev_val_data": pd.DataFrame()
    }

    kf_all_path = Path(Path(kf_rank).joinpath("all_features"))
    write_training_files(features, kf_all_path, "kid-friend", all_feats=True)

    # rename the file to avoid overwriting conflicts
    current_path = Path(kf_all_path).joinpath(f"kid-friend_test_data.txt")
    new_path = Path(kf_all_path).joinpath(f"kid-friend_google_test_data.txt")
    current_path.rename(new_path)


def process_kid_friend_bm25():
    kf_rank = Path(__file__).parent.parent.joinpath(
        "datasets", "ranking", "data", "kid-friend")
    kf_parent = Path(__file__).parent.parent.joinpath(
        "datasets", "ranking", "data", "test_data", "test_data", "kid-friend")
    kf_docs_path = Path(__file__).parent.parent.joinpath(
        "datasets", "ranking", "data", "test_data", "test_data", "kid-friend", "corpus", "documents.jsonl")
    # kf_qrels_path = Path(__file__).parent.parent.joinpath(
    #     "datasets", "ranking", "data", "test_data", "test_data", "kid-friend", "qrels", "qrels-relevance-binary.csv")
    # qrels = pd.read_csv(kf_qrels_path)
    ranked = pd.read_csv(Path(kf_rank).joinpath("kid_friend_BM25.res"), names=["qid", "q0", "docno", "rank", "bm25_score", "ranker"], sep=" ")
    docs = pd.read_json(kf_docs_path, lines=True)
    docs.rename(columns={"id": "docno"}, inplace=True)
    combined = ranked.copy()
    combined["snippet"] = combined.apply(get_text, args=(docs,), axis=1)
    combined.rename(columns={"docno": "doc_id"}, inplace=True)
    combined.to_csv(Path(kf_parent).joinpath("bm25-with-docs.csv"), index=False)

    # Process the documents for features to be written into a svm_light file
    data = feature_generation(combined)

    features = {
        "train_data": pd.DataFrame(),
        "test_data": data,
        "val_data": pd.DataFrame(),
        "dev_train_data": pd.DataFrame(),
        "dev_test_data": pd.DataFrame(),
        "dev_val_data": pd.DataFrame()
    }

    kf_all_path = Path(Path(kf_rank).joinpath("all_features"))
    write_training_files(features, kf_all_path, "kid-friend", all_feats=True)

    # rename the file to avoid overwriting conflicts
    Path(kf_all_path).joinpath(f"kid-friend_test_data.txt").rename("kid-friend_bm25_test_data.txt")


def process_newsela():
    redorank_dir = Path(DATA_DIR).joinpath("ranking", "data", "redorank_2026")
    adarank_dir = Path(DATA_DIR).joinpath("ranking", "data", "adarank_2026")
    redorank_dir.mkdir(parents=True, exist_ok=True)
    adarank_dir.mkdir(parents=True, exist_ok=True)
    data_file = Path(DATA_DIR).joinpath("ranking", "data", "newsela", "newsela_cranfield.csv")
    newsela = pd.read_csv(data_file)

    # Generate features and divide data into train, test, val
    data = feature_generation(newsela)
    qids = list(set(data.qid.values))
    train_qids = random.sample(qids, round(len(qids) * 0.8))  # 80% of data is train data
    test_qids = list(set(qids) - set(train_qids))  # Remaining 20% is test data
    val_qids = random.sample(train_qids, round(len(train_qids) * 0.1))
    train_qids = list(set(train_qids) - set(val_qids))
    dev_qids = random.sample(train_qids, round(len(train_qids) * 0.4))
    train_qids = list(set(train_qids) - set(dev_qids))

    # Split dev into train and test now
    dev_train_qids = random.sample(dev_qids, round(len(dev_qids) * 0.8))
    dev_test_qids = list(set(dev_qids) - set(dev_train_qids))
    dev_val_qids = random.sample(dev_train_qids, round(len(dev_train_qids) * 0.2))
    dev_train_qids = list(set(dev_train_qids) - set(dev_val_qids))

    train_data = data[data["qid"].isin(train_qids)]
    test_data = data[data["qid"].isin(test_qids)]
    val_data = data[data["qid"].isin(val_qids)]
    dev_train_data = data[data["qid"].isin(dev_train_qids)]
    dev_test_data = data[data["qid"].isin(dev_test_qids)]
    dev_val_data = data[data["qid"].isin(dev_val_qids)]

    features = {
        "train_data": train_data,
        "test_data": test_data,
        "val_data": val_data,
        "dev_train_data": dev_train_data,
        "dev_test_data": dev_test_data,
        "dev_val_data": dev_val_data
    }

    redo_edu_dir = Path(redorank_dir).joinpath("educational")
    write_training_files(features, redo_edu_dir, "redorank_2026", edu_only=True)
    ada_edu_dir = Path(adarank_dir).joinpath("educational")
    write_training_files(features, ada_edu_dir, "adarank_2026", edu_only=True)

    redo_read_dir = Path(redorank_dir).joinpath("readability")
    write_training_files(features, redo_read_dir, "redorank_2026", read_only=True)
    ada_read_dir = Path(adarank_dir).joinpath("educational")
    write_training_files(features, ada_read_dir, "adarank_2026", read_only=True)

    redo_obj_dir = Path(redorank_dir).joinpath("objectionable")
    write_training_files(features, redo_obj_dir, "redorank_2026", obj_only=True)
    ada_obj_dir = Path(adarank_dir).joinpath("objectionable")
    write_training_files(features, ada_obj_dir, "adarank_2026", obj_only=True)

    # For all features, write them to both AdaRank and REdORank directories
    redo_all_dir = Path(redorank_dir).joinpath("all_features")
    write_training_files(features, redo_all_dir, "redorank_2026", all_feats=True)
    ada_all_dir = Path(adarank_dir).joinpath("all_features")
    write_training_files(features, ada_all_dir, "adarank_2026", all_feats=True)


if __name__ == "__main__":
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument("--requik_bm25", action="store_true",
                    help="Flag to indicate running the process of ReQuik data using scores from BM25.")
    ap.add_argument("--kid_friend_bm25", action="store_true",
                    help="Flag to indicate running the process of KidFriend data using scores from BM25.")
    ap.add_argument("--kid_friend_bing", action="store_true",
                    help="Flag to indicate running the process of KidFriend data using scores from Bing.")
    ap.add_argument("--kid_friend_google", action="store_true",
                    help="Flag to indicate running the process of KidFriend data using scores from Google.")
    ap.add_argument("--newsela", action="store_true",
                    help="Flag to indicate running the process of Newsela data.")
    args = ap.parse_args()

    if args.requik_bm25:
        process_requik_bm25()

    if args.kid_friend_bm25:
        process_kid_friend_bm25()

    if args.kid_friend_bing:
        process_kid_friend_bing()

    if args.kid_friend_google:
        process_kid_friend_google()

    if args.newsela:
        process_newsela()
