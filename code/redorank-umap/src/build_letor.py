"""
This script creates the training, test, and a validation files for REdORank following the SVM-Rank style.
Reference: https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html

NOTE: The execution of the process_newsela.py script must precede this one.
"""
import pandas as pd
import random
from objectionable.ObjectionabilityEstimator import ObjectionabilityEstimator
from pathlib import Path
from readability.formulas import spache_allen
from tqdm import tqdm


DATA_DIR = Path(__file__).resolve().parent.parent.joinpath("datasets")


def build_data_files():
    """
    Constructs the train, test, and validation files for REdORank and AdaRank models using the Edu, Read, and Obj
    features. Produces two directories of data files -- redorank and adarank.

    Returns
    -------
    None
    """
    redorank_dir = Path(DATA_DIR).joinpath("ranking", "data", "redorank")
    adarank_dir = Path(DATA_DIR).joinpath("ranking", "data", "adarank")
    redorank_dir.mkdir(parents=True, exist_ok=True)
    adarank_dir.mkdir(parents=True, exist_ok=True)
    data_file = Path(DATA_DIR).joinpath("ranking", "data", "newsela", "newsela_cranfield.csv")
    newsela = pd.read_csv(data_file)
    newsela.rename(columns={"snippet": "description"}, inplace=True)

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

    data = {
        "train": train_data,
        "test": test_data,
        "val": val_data,
        "dev_train": dev_train_data,
        "dev_test": dev_test_data,
        "dev_val": dev_val_data
    }

    redo_edu_dir = Path(redorank_dir).joinpath("educational")
    write_training_files(data, redo_edu_dir, "redorank", edu_only=True)
    ada_edu_dir = Path(adarank_dir).joinpath("educational")
    write_training_files(data, ada_edu_dir, "adarank", edu_only=True)

    redo_read_dir = Path(redorank_dir).joinpath("readability")
    write_training_files(data, redo_read_dir, "redorank", read_only=True)
    ada_read_dir = Path(adarank_dir).joinpath("educational")
    write_training_files(data, ada_read_dir, "adarank", read_only=True)

    redo_obj_dir = Path(redorank_dir).joinpath("objectionable")
    write_training_files(data, redo_obj_dir, "redorank", obj_only=True)
    ada_obj_dir = Path(adarank_dir).joinpath("objectionable")
    write_training_files(data, ada_obj_dir, "adarank", obj_only=True)

    # For all features, write them to both AdaRank and REdORank directories
    redo_all_dir = Path(redorank_dir).joinpath("all_features")
    write_training_files(data, redo_all_dir, "redorank", all_feats=True)
    ada_all_dir = Path(adarank_dir).joinpath("all_features")
    write_training_files(data, ada_all_dir, "adarank", all_feats=True)


def write_training_files(data, directory, model_name, edu_only=False, read_only=False, obj_only=False, all_feats=False):
    """
    Writes features to train, test, and val files in same format as LETOR dataset.

    Parameters
    ----------
    data
        DataFrame containing feature values to be written to LETOR files.
    directory
        Path object representing location for output.
    model_name
        Name of the model the dataset is being built for. Options are 'adarank' and 'redorank'.
    edu_only
        When set, LETOR file will only have educational feature value.
    read_only
        When set, LETOR file will only have readability feature value.
    obj_only
        When set, LETOR file will only have objectionable feature value.
    all_feats
        When set, LETOR file will have all feature values.

    Returns
    -------
    None
    """
    directory.mkdir(parents=True, exist_ok=True)
    train_output_filename = Path(directory).joinpath(f"{model_name}_train_data.txt")
    test_output_filename = Path(directory).joinpath(f"{model_name}_test_data.txt")
    val_output_filename = Path(directory).joinpath(f"{model_name}_val_data.txt")
    dev_train_output_filename = Path(directory).joinpath(f"{model_name}_dev_train_data.txt")
    dev_test_output_filename = Path(directory).joinpath(f"{model_name}_dev_test_data.txt")
    dev_val_output_filename = Path(directory).joinpath(f"{model_name}_dev_val_data.txt")

    if not data["train_data"].empty:
        to_letor(data["train_data"], train_output_filename, edu_only=edu_only, read_only=read_only, obj_only=obj_only,
                 all_feats=all_feats)
    if not data["test_data"].empty:
        to_letor(data["test_data"], test_output_filename, edu_only=edu_only, read_only=read_only, obj_only=obj_only,
                 all_feats=all_feats)
    if not data["val_data"].empty:
        to_letor(data["val_data"], val_output_filename, edu_only=edu_only, read_only=read_only, obj_only=obj_only,
                 all_feats=all_feats)
    if not data["dev_train_data"].empty:
        to_letor(data["dev_train_data"], dev_train_output_filename, edu_only=edu_only, read_only=read_only,
                 obj_only=obj_only, all_feats=all_feats)
    if not data["dev_test_data"].empty:
        to_letor(data["dev_test_data"], dev_test_output_filename, edu_only=edu_only, read_only=read_only, obj_only=obj_only,
                 all_feats=all_feats)
    if not data["dev_val_data"].empty:
        to_letor(data["dev_val_data"], dev_val_output_filename, edu_only=edu_only, read_only=read_only, obj_only=obj_only,
                 all_feats=all_feats)


def feature_generation(content_frame):
    """
    Generates the Edu, Read, and Obj features for a collection of resources.

    Parameters
    ----------
    content_frame
        Pandas DataFrame of resources for which the features will be generated.
        Required column names: description

    Returns
    -------
    DataFrame containing the original data as well as the new feature columns -- edu_pred, non_edu_pred, edu_prob,
    reading_level, obj_prob.
    """
    # Educational
    print("Calculating probability of educational alignment")
    if ("bigbert_pred" in content_frame.columns and "bigbert_prob_non_edu" in content_frame.columns and
            "bigbert_prob_edu" in content_frame.columns):
        content_frame.rename(columns={"bigbert_pred": "edu_pred", "bigbert_prob_non_edu": "non_edu_prob",
                                      "bigbert_prob_edu": "edu_prob"}, inplace=True)
    elif ("edu_pred" in content_frame.columns and "non_edu_prob" in content_frame.columns and
          "edu_prob" in content_frame.columns):
        pass
    else:
        bigbert_data = pd.read_csv(Path(DATA_DIR).joinpath("educational", "data",
                                                           "bigbert_newsela_cranfield_predictions.csv"))

        content_frame["edu_pred"] = bigbert_data["bigbert_pred"]
        content_frame["non_edu_prob"] = bigbert_data["bigbert_prob_non_edu"]
        content_frame["edu_prob"] = bigbert_data["bigbert_prob_edu"]

    # Readability
    tqdm.pandas(desc="Calculating readability level", unit=" resources")
    content_frame["reading_level"] = content_frame["description"].progress_apply(spache_allen)

    # Objectionable
    print("Training the objectionable prediction model")
    train_data_dir = str(Path(DATA_DIR).joinpath("objectionable", "data"))
    train_df = pd.read_csv(Path(train_data_dir).joinpath("redorank_obj_train_set.csv"))
    train_data = train_df.drop_duplicates(subset=["content"])
    oc = ObjectionabilityEstimator()
    oc.fit(train_data)
    print("Training completed!")

    tqdm.pandas(desc="Determining probability of being objectionable", unit=" resources")
    content_frame["obj_prob"] = content_frame.progress_apply(lambda x: oc.predict_proba(x["description"]), axis=1)

    return content_frame


def to_letor(frame, output_filename, edu_only=False, read_only=False, obj_only=False, all_feats=False):
    """
    For details on the data format, see: https://arxiv.org/ftp/arxiv/papers/1306/1306.2597.pdf
    """
    rows = frame.shape[0]
    qids = frame["qid"].tolist()
    edu_feat = frame["edu_prob"].tolist()
    read_feat = frame["reading_level"].tolist()
    obj_feat = frame["obj_prob"].tolist()
    rank = frame["rank"].tolist()
    if "doc_id" in frame.columns:
        doc_ids = frame["doc_id"].tolist()
    else:
        doc_ids = None

    with open(str(output_filename), "w") as outfile:
        for row in range(rows):
            target = rank[row]

            # if row == 0 or qids[row] != qids[row - 1]:
            #     target = 2
            # elif row == (len(rank) - 1) or qids[row] != qids[row + 1]:
            #     target = 0
            # else:
            #     target = 1

            if edu_only:
                line = f"{target} qid:{qids[row]} 1:{edu_feat[row]}\n"
                if Path(output_filename).parent.parent.name == "redorank":
                    line = f"{target} qid:{qids[row]} 1:{edu_feat[row]} 2:{obj_feat[row]}\n"
            if read_only:
                line = f"{target} qid:{qids[row]} 1:{read_feat[row]}\n"
                if Path(output_filename).parent.parent.name == "redorank":
                    line = f"{target} qid:{qids[row]} 1:{read_feat[row]} 2:{obj_feat[row]}\n"
            if obj_only:
                line = f"{target} qid:{qids[row]} 1:{obj_feat[row]}\n"
            if all_feats:
                if doc_ids is not None:
                    line = f"{target} qid:{qids[row]} 1:{edu_feat[row]} 2:{read_feat[row]} 3:{obj_feat[row]} 4:{doc_ids[row]}\n"
                else:
                    line = f"{target} qid:{qids[row]} 1:{edu_feat[row]} 2:{read_feat[row]} 3:{obj_feat[row]}\n"

            outfile.write(line)

    print(f"{rows} lines written.")


if __name__ == "__main__":
    print("Building data files")
    build_data_files()
