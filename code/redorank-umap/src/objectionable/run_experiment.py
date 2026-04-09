import pandas as pd
import re
import time
from awessome.awessome_builder import *
from baselines.naive_bayes_classifier import run_model, _tokenize_and_clean, make_unique
from flair.data import Sentence
from flair.models import TextClassifier
from korsce.KSAppropriateness import AppropriatenessEstimation
from korsce.ObjectionabilityEstimator import ObjectionabilityEstimator
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from notebooks.awessome_objectionable import score_passage_and_predict
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def read_words_list(filepath):
    words_frame = pd.read_csv(filepath, names=["words"])
    words_list = words_frame.words.to_list()
    return words_list


def gather_data_set(train_file, test_file):
    # Load the data
    dataset_dir = str(Path(__file__).resolve().parent.parent.parent.joinpath("datasets", "objectionable"))
    train_df = pd.read_csv(Path(dataset_dir).joinpath(train_file))
    test_df = pd.read_csv(Path(dataset_dir).joinpath(test_file))
    # test_df.rename(columns={"objectionable": "content_label"}, inplace=True)
    # test_df.reset_index(inplace=True, drop=True)

    tqdm.pandas(desc="Pre-processing training content")
    # train_df["pre_proc_content"] = train_df["content"].progress_apply(_tokenize_and_clean)
    train_corpus = train_df.drop_duplicates(subset=["content"])

    tqdm.pandas(desc="Pre-processing test content")
    # test_df["pre_proc_content"] = test_df["content"].progress_apply(_tokenize_and_clean)
    test_corpus = test_df.drop_duplicates(subset=["content"])

    # Ensure no cross-contamination between sets
    remove_these = make_unique(test_corpus["content"], train_corpus["content"])
    train_corpus = train_corpus[~train_corpus["content"].isin(remove_these)]

    return train_corpus, test_corpus


def build_awessome_lexicon(content, output_filepath, weight):
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(content)

    terms = ["abort", "alcohol", "tobacco", "illegal", "affair", "drug", "gamb", "marij", "porn", "violen", "racism",
             "weapon"]
    lexicon = []
    n = int(round(len(finder.score_ngrams(bigram_measures.pmi)) * 0.01, 0))
    for term in tqdm(terms):
        finder.apply_ngram_filter(lambda *w: not len(re.findall(r'\b{}\w+'.format(term), ' '.join(w))) > 0)
        pmi = finder.nbest(bigram_measures.pmi, n)
        llg = finder.nbest(bigram_measures.likelihood_ratio, n)
        n = int(round(len(finder.score_ngrams(bigram_measures.pmi)) * 0.01, 0))
        top_1_word = []
        for x in range(n):
            for y in pmi[x]:
                if not y.startswith(term):
                    top_1_word.append(y)
                    break
            for z in llg[x]:
                if not z.startswith(term):
                    top_1_word.append(z)
                    break

        top_1_word = list(set(top_1_word))
        lexicon.extend(top_1_word)

    with open(Path(dataset_dir).joinpath("awessome", output_filepath), "w") as outfile:
        for l in tqdm(lexicon, desc="Writing AWESSOME lexicon"):
            outfile.write(f"{l}\t{weight}\n")


def run_awessome(val_data):
    dataset_dir = str(Path(__file__).resolve().parent.parent.parent.joinpath("datasets", "objectionable"))
    avg_builder = SentimentIntensityScorerBuilder('avg', 'bert-base-nli-mean-tokens', 'cosine', '100', True)
    avg_scorer = avg_builder.build_scorer_from_lexicon_file(Path(dataset_dir).joinpath("awessome",
                                                                                       "awessome_full_lexicon.txt"))

    tqdm.pandas(desc="Scoring at passage level...")
    scores, predictions = val_data.progress_apply(lambda x: score_passage_and_predict(avg_scorer, x["content"]), axis=1,
                                                  result_type="expand")
    return scores, predictions


def run_korsce_appropriateness(train_data, val_data):
    ae = AppropriatenessEstimation(clf="random_forest")
    ae.fit(train_data)

    tqdm.pandas(desc="Predicting the documents with KSApp")
    probs, predictions = val_data.progress_apply(lambda x: ae.predict(x["content"], "", ""), axis=1,
                                                 result_type="expand")

    return probs, predictions


def run_naive_bayes(train_data, val_data):
    probs, predictions = run_model(train_data, val_data)
    return probs, predictions


def run_bert_for_tc(val_data):
    # Perform the prediction
    print("Making prediction on test corpus")
    classifier = TextClassifier.load(Path(dataset_dir).resolve().joinpath(
        "resources", "bert_obj_tclf", "final-model.pt"))

    sentences = val_data["content"].tolist()
    sentences_cls = [Sentence(x) for x in sentences]

    [classifier.predict(s) for s in sentences_cls]
    [s.labels.sort() for s in sentences_cls]

    predictions = [1 if "OBJ" in str(s.labels.pop()) else 0 for s in tqdm(sentences_cls,
                                                                          desc="Assigning labels from BERT")]
    return predictions


def run_objectionability_classifier(train_data, val_data):
    oc = ObjectionabilityEstimator()
    oc.fit(train_data)

    val_data[oc.features] = val_data.progress_apply(
        lambda x: oc.generate_features(x["content"], train=True), axis=1, result_type="expand")

    tqdm.pandas(desc="Predicting class with ObjEst")
    predictions = val_data.progress_apply(lambda x: oc.predict(x["content"]), axis=1)
    predictions = [int(pred) for pred in predictions]
    tqdm.pandas(desc="Predicting probabilities with ObjEst")
    probs = val_data.progress_apply(lambda x: oc.predict_proba(x["content"]), axis=1)

    val_data.to_csv(Path(dataset_dir).joinpath("oc_validation_data_with_features.csv"), index=False)

    return probs, predictions, val_data


if __name__ == "__main__":
    tqdm.pandas()
    dataset_dir = str(Path(__file__).resolve().parent.parent.parent.joinpath("datasets", "objectionable"))
    # train_data_version = "dev_train"
    # val_data_version = "dev_test"
    train_data_version = "train"
    val_data_version = "test"

    # Load the data
    val_data_file = f"redorank_obj_{val_data_version}_set.csv"
    train_data_file = f"redorank_obj_{train_data_version}_set.csv"

    train_data, val_data = gather_data_set(train_data_file, val_data_file)

    # ReDORank's Objectionability Estimator
    probs, predictions, val_data = run_objectionability_classifier(train_data, val_data)
    val_data["ap_probs"] = probs
    val_data["ap_prediction"] = predictions

    # Naive Bayes classifier
    probs, predictions = run_naive_bayes(train_data, val_data)
    val_data["nb_probs"] = probs
    val_data["nb_prediction"] = predictions

    # AWESSOME
    awe_scores, awe_preds = run_awessome(val_data)
    val_data["awessome_prediction"] = awe_preds
    val_data["awessome_score"] = awe_scores

    # Korsce Appropriateness
    ks_app_probs, ks_app_preds = run_korsce_appropriateness(train_data, val_data)
    val_data["ks_app_probs"] = ks_app_probs
    val_data["ks_app_prediction"] = ks_app_preds

    # BERT Classification
    val_data["bert_4_tc_prediction"] = run_bert_for_tc(val_data)

    # Analyze results
    results = [
               {"model": "Objectionablity Estimator",
                "accuracy": accuracy_score(val_data["label"], val_data["ap_prediction"])},
               {"model": "Multinomial Naive-Bayes",
                "accuracy": accuracy_score(val_data["label"], val_data["nb_prediction"])},
               {"model": "AWESSOME",
                "accuracy": accuracy_score(val_data["label"], val_data["awessome_prediction"])},
               {"model": "KSAppropriateness",
                "accuracy": accuracy_score(val_data["label"], val_data["ks_app_prediction"])},
               {"model": "BERT4TC",
                "accuracy": accuracy_score(val_data["label"], val_data["bert_4_tc_prediction"])},
               {"model": "Majority Classifier",
                "accuracy": accuracy_score(val_data["label"],
                                           [val_data["label"].value_counts().idxmax()] * val_data.shape[0])}
               ]

    results_frame = pd.DataFrame(results)
    print(results_frame)

    results_frame.to_csv(Path(dataset_dir).joinpath("model_comparison_results.csv"), index=False)
    val_data.to_csv(Path(dataset_dir).joinpath("validation_data_with_predictions.csv"), index=False)
