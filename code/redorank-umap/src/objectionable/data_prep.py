"""
Script to construct the training data files for Judge_bad classifier
"""
import pandas as pd
from nltk.corpus import stopwords
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATASET_DIR = Path(__file__).resolve().parent.parent.parent.joinpath("datasets")
OBJ_DIR = Path(DATASET_DIR).joinpath("objectionable", "data")
EDU_DIR = Path(DATASET_DIR).joinpath("educational", "data")
VOCAB_DIR = Path(DATASET_DIR).joinpath("objectionable", "data", "vocab_files")


def display_dataset_details(redorank_obj_rf_train_set, alexa_adult, alexa_non_obj):
    """
    Function to output details of the constructed dataset to the screen.

    Parameters
    ----------
    redorank_obj_rf_train_set
        Full dataset to be used for training the Judge_bad classifier.
    alexa_adult
        Collection of adult/objectionable resources from the Alexa Top Sites by Category dataset.
    alexa_non_obj
        Collection of non-objectionable  resources from the Alexa Top Sites by Category dataset.

    Returns
    -------
    None
    """
    print(f"ALEXA OBJ\t({alexa_adult.shape[0]} resources)")
    print(f"ALEXA NON-OBJ\t({alexa_non_obj.shape[0]} resources)")
    print("\n\n")

    print("COMBINED")
    print("=" * 75)
    print(f"Objectionable Resources:\t\t{redorank_obj_rf_train_set[redorank_obj_rf_train_set['label'] == 1].shape[0]}\n"
          f"Non-Objectionable Resources:\t\t"
          f"{redorank_obj_rf_train_set[redorank_obj_rf_train_set['label'] == 0].shape[0]}\n"
          f"Total Resources:\t\t\t{redorank_obj_rf_train_set.shape[0]}")


def build_safe_and_unsafe():
    """
    Function that selects non-objectionable resources from Alexa Top Sites by Category and combines with objectionable
    ones to form a complete dataset. Produces the datafile redorank_obj_set.csv.

    Returns
    -------
    None
    """
    # ALEXA Data

    # Objectionable Alexa websites
    alexa_adult = pd.read_csv(Path(OBJ_DIR).joinpath("alexa", "adult_sites.csv"))
    alexa_adult["content"] = alexa_adult.apply(lambda x: f"{x['title']}. {x['description']}", axis=1)
    alexa_adult.rename(columns={"pornography": "label"}, inplace=True)

    # Full Alexa data, that we grab a selection of those that are not in the above, and not educational
    alexa_edu_categories = ['Top/Kids_and_Teens/Pre-School', 'Top/Kids_and_Teens/School_Time']
    alexa_adult_categories = list(alexa_adult.category.unique())
    excluded_cats = alexa_edu_categories + alexa_adult_categories

    alexa_full = pd.read_csv(Path(OBJ_DIR).joinpath("alexa", "alexa_full.csv"))
    alexa_usable = alexa_full[~alexa_full["category"].isin(excluded_cats)]

    non_obj_sample_size = alexa_adult.shape[0] * 4
    alexa_non_obj = alexa_usable.sample(non_obj_sample_size, random_state=42)
    alexa_non_obj["label"] = 0

    # Normalize columns before combining -- URL, Content, Label
    alexa_adult = alexa_adult[["url", "content", "label"]]
    alexa_non_obj = alexa_non_obj[["url", "content", "label"]]

    # Join safe and unsafe into a training file
    redorank_obj_rf_train_set = pd.concat([alexa_adult, alexa_non_obj])
    redorank_obj_rf_train_set.to_csv(Path(OBJ_DIR).joinpath("redorank_obj_set.csv"),
                                     index=False)

    display_dataset_details(redorank_obj_rf_train_set, alexa_adult, alexa_non_obj)


def build_train_test_sets():
    """
    Function to split ObjSet into train, test, and dev train/test sets. Prints information about each set created and
    produces one file for each set.

    Returns
    -------
    None
    """
    full_set = pd.read_csv(Path(OBJ_DIR).joinpath("redorank_obj_set.csv"))
    x_train, x_test = train_test_split(full_set, test_size=0.5, random_state=42)
    x_train, x_dev = train_test_split(x_test, test_size=0.5, random_state=42)

    x_dev_train, x_dev_test = train_test_split(x_dev, test_size=0.2, random_state=42)

    x_train.to_csv(Path(OBJ_DIR).joinpath("redorank_obj_train_set.csv"), index=False)
    x_test.to_csv(Path(OBJ_DIR).joinpath("redorank_obj_test_set.csv"), index=False)
    x_dev_train.to_csv(Path(OBJ_DIR).joinpath("redorank_obj_dev_train_set.csv"), index=False)
    x_dev_test.to_csv(Path(OBJ_DIR).joinpath("redorank_obj_dev_test_set.csv"), index=False)

    print("\n\nTRAIN SET")
    print("=" * 75)
    print(f"OBJ:\t\t{x_train[x_train['label'] == 1].shape[0]}")
    print(f"NON-OBJ:\t\t{x_train[x_train['label'] == 0].shape[0]}")
    print(f"TOTAL:\t\t{x_train.shape[0]}")
    print("\n\n")

    print("TEST SET\n")
    print("=" * 75)
    print(f"OBJ:\t\t{x_test[x_test['label'] == 1].shape[0]}")
    print(f"NON-OBJ:\t\t{x_test[x_test['label'] == 0].shape[0]}")
    print(f"TOTAL:\t\t{x_test.shape[0]}")


def get_files(directory_path):
    """
    Simple retrieval function to gather all objectionable vocabulary files
    Parameters
    ----------
    directory_path
        Path to directory containing the vocabulary files.

    Returns
    -------
    List of Path objects for each vocabulary file.
    """
    files = []
    for x in directory_path.iterdir():
        if x.is_file():
            files.append(x)
    return files


def remove_stopwords():
    """
    Function to remove stop words from each objectionable vocabulary file.

    Returns
    -------
    None
    """
    stop_words = set(stopwords.words("english"))
    files = get_files(Path(VOCAB_DIR))
    for f in tqdm(files):
        df = pd.read_csv(f, names=["term"])
        sw_removed = [w for w in df["term"].values if not w.lower() in stop_words]
        new_df = pd.DataFrame(sw_removed)
        new_df.to_csv(f, index=False)


if __name__ == "__main__":
    build_safe_and_unsafe()
    build_train_test_sets()
    remove_stopwords()
