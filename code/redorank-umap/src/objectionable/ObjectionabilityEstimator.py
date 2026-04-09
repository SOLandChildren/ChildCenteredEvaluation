import pandas as pd
import nltk
import re
import string
import time
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import enchant
from collections import Counter
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tqdm import tqdm


DATASET_DIR = str(Path(__file__).resolve().parent.parent.parent.parent.joinpath("datasets", "objectionable"))


class ObjectionabilityEstimator(object):
    # Initializer / Instance Attributes
    def __init__(self):
        self.train_file = "redorank_obj_est_training_features.json"
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.sexually_explicit_words = self._read_words_list(
            Path(DATASET_DIR).joinpath("korsce", "appropriateness", "sexually_explicit_words.csv"))
        self.hate_speech_words = self._read_words_list(
            Path(DATASET_DIR).joinpath("korsce", "appropriateness", "hate_speech_words.csv"))
        self.abortion_words = self._read_words_list(
            Path(DATASET_DIR).joinpath("vocab_files", "abortion_words_list.csv"))
        self.alcohol_and_tobacco_words = self._read_words_list(
            Path(DATASET_DIR).joinpath("vocab_files", "alcohol_and_tobacco_words_list.csv"))
        self.drugs_words = self._read_words_list(Path(DATASET_DIR).joinpath("vocab_files", "drugs_words_list.csv"))
        self.gambling_words = self._read_words_list(Path(DATASET_DIR).joinpath(
            "vocab_files", "gambling_words_list.csv"))
        self.illegal_affairs_words = self._read_words_list(
            Path(DATASET_DIR).joinpath("vocab_files", "illegal_affairs_words_list.csv"))
        self.marijuana_words = self._read_words_list(
            Path(DATASET_DIR).joinpath("vocab_files", "marijuana_words_list.csv"))
        self.violence_words = self._read_words_list(
            Path(DATASET_DIR).joinpath("vocab_files", "violence_words_list.csv"))
        self.weapons_words = self._read_words_list(
            Path(DATASET_DIR).joinpath("vocab_files", "weapons_words_list.csv"))
        self.features = ["inap_prev_doc", "inap_cov_doc", "misp_prev_doc", "misp_cov_doc", "hate_prev_doc",
                         "hate_cov_doc", "abortion_prev_doc", "abortion_cov_doc", "ia_prev_doc", "ia_cov_doc",
                         "drugs_prev_doc", "drugs_cov_doc", "gam_prev_doc", "gam_cov_doc", "vio_prev_doc",
                         "vio_cov_doc"]
        self.d = enchant.Dict("en_US")
        try:
            self.stops = stopwords.words('english')
        except LookupError:
            nltk.download("stopwords")
            self.stops = stopwords.words('english')
        self.classifier = RandomForestClassifier(
            criterion='gini', max_depth=64, max_leaf_nodes=32, min_samples_leaf=2, min_samples_split=64)


    def _inappropriate_term_frequency(self, counter, lexicon):
        inap_terms = self._get_inappropriate_words(counter, lexicon)
        inap_tf = sum([counter[t] for t in inap_terms])
        return inap_tf

    def _read_words_list(self, filepath):
        words_frame = pd.read_csv(filepath, names=["words"])
        words_list = words_frame.words.to_list()
        corrected = [self.lemmatizer.lemmatize(s, self._get_wordnet_pos(s)).strip() for s in words_list]

        return corrected

    @staticmethod
    def _get_inappropriate_words(counter, lexicon):
        terms = counter.keys()
        inap_terms = [x for x in terms for inap_term in lexicon
                      if x == inap_term]
        return inap_terms

    def _get_misspelled_words(self, words):
        misspelled = [w for w in words if self.d.check(w) is False]  # Total
        return misspelled

    @staticmethod
    def _get_wordnet_pos(word):
        try:
            tag = nltk.pos_tag([word])[0][1][0].upper()
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
            tag = nltk.pos_tag([word])[0][1][0].upper()
        try:
            tag_dict = {
                "J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV
            }
        except LookupError:
            nltk.download("wordnet")
            tag_dict = {
                "J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV
            }

        return tag_dict.get(tag, wordnet.NOUN)

    def _tokenize_and_clean(self, content):
        tokens = [nltk.word_tokenize(content.lower().replace("–", " "))]
        flattened_tokens = [s for k in tokens for s in k]
        flattened_tokens = [re.sub(r'[^\x00-\x7F]+', '', s) for s in flattened_tokens]
        flattened_tokens = [s for s in flattened_tokens if s]
        sw_removed = [w for w in flattened_tokens if not w.lower() in self.stops]
        stemmed_tokens = [self.lemmatizer.lemmatize(s, self._get_wordnet_pos(s)).strip() for s in sw_removed]
        cleaned_tokens = [''.join(c for c in s if c not in string.punctuation) for s in stemmed_tokens]
        cleaned_tokens = [w for w in cleaned_tokens if w and len(w) in range(2, 15)]
        return cleaned_tokens

    def inap_prevalence(self, counter, lexicon):
        inap_term_count = self._inappropriate_term_frequency(counter, lexicon)
        normalization_factor = sum(counter.values())
        if normalization_factor == 0:
            return 0.0
        return inap_term_count / normalization_factor
    
    def inap_coverage(self, counter, lexicon):
        inap_words = self._get_inappropriate_words(counter, lexicon)
        unique_inap_terms = list(set(inap_words))
        score = len(unique_inap_terms) / len(lexicon)
        return score
    
    def normalize_lexicon_length(self, lexicon):
        lex_lens = [len(self.sexually_explicit_words), len(self.hate_speech_words), len(self.abortion_words), 
                    len(self.drugs_words), len(self.alcohol_and_tobacco_words), len(self.gambling_words), 
                    len(self.illegal_affairs_words), len(self.marijuana_words), len(self.violence_words), 
                    len(self.weapons_words)]
        x_min = min(lex_lens)
        x_max = max(lex_lens)
        x_prime = (len(lexicon) - x_min) * 10 / (x_max - x_min)
        return x_prime

    def misspelled_prevalence(self, resource):
        misspelled = self._get_misspelled_words(resource)
        counter = Counter(resource)
        normalization_factor = sum(counter.values())
        if normalization_factor == 0:
            return 0.0
        return len(misspelled) / normalization_factor

    def misspelled_coverage(self, resource):
        misspelled = self._get_misspelled_words(resource)
        unique_misspelled = list(set(misspelled))
        obj_lexicon = set(list(self.sexually_explicit_words + self.hate_speech_words + self.abortion_words +
                               self.drugs_words + self.alcohol_and_tobacco_words + self.gambling_words +
                               self.illegal_affairs_words + self.marijuana_words + self.violence_words +
                               self.weapons_words))
        misspelled_inap = [x for x in unique_misspelled if x in obj_lexicon]

        return len(misspelled_inap) / len(obj_lexicon)
    
    def get_feature_names(self):
        return self.features
    
    def generate_features(self, resource, train=False):
        cleaned_resource = self._tokenize_and_clean(resource)
        term_count = Counter(cleaned_resource)
        drugs = []
        drugs.extend(self.drugs_words)
        drugs.extend(self.marijuana_words)
        drugs.extend(self.alcohol_and_tobacco_words)
        violence_and_weapons = []
        violence_and_weapons.extend(self.violence_words)
        violence_and_weapons.extend(self.weapons_words)

        inap_prev_doc = self.inap_prevalence(term_count, self.sexually_explicit_words)
        inap_cov_doc = self.inap_coverage(term_count, self.sexually_explicit_words)
        hate_prev_doc = self.inap_prevalence(term_count, self.hate_speech_words)
        hate_cov_doc = self.inap_coverage(term_count, self.hate_speech_words)
        abortion_prev_doc = self.inap_prevalence(term_count, self.abortion_words)
        abortion_cov_doc = self.inap_coverage(term_count, self.abortion_words)
        ia_prev_doc = self.inap_prevalence(term_count, self.illegal_affairs_words)
        ia_cov_doc = self.inap_coverage(term_count, self.illegal_affairs_words)
        drugs_prev_doc = self.inap_prevalence(term_count, drugs)
        drugs_cov_doc = self.inap_coverage(term_count, drugs)
        gam_prev_doc = self.inap_prevalence(term_count, self.gambling_words)
        gam_cov_doc = self.inap_coverage(term_count, self.gambling_words)
        vio_prev_doc = self.inap_prevalence(term_count, violence_and_weapons)
        vio_cov_doc = self.inap_coverage(term_count, violence_and_weapons)

        misp_prev_doc = self.misspelled_prevalence(cleaned_resource)
        misp_cov_doc = self.misspelled_coverage(cleaned_resource)

        if train:
            return (inap_prev_doc, inap_cov_doc, misp_prev_doc, misp_cov_doc, hate_prev_doc,
                    hate_cov_doc, abortion_prev_doc, abortion_cov_doc, ia_prev_doc, ia_cov_doc,
                    drugs_prev_doc, drugs_cov_doc, gam_prev_doc, gam_cov_doc, vio_prev_doc, vio_cov_doc)
        else:
            return [inap_prev_doc, inap_cov_doc, misp_prev_doc, misp_cov_doc, hate_prev_doc,
                    hate_cov_doc, abortion_prev_doc, abortion_cov_doc, ia_prev_doc, ia_cov_doc,
                    drugs_prev_doc, drugs_cov_doc, gam_prev_doc, gam_cov_doc, vio_prev_doc, vio_cov_doc]

    def _generate_training_file(self, training_data, features_output="redorank_obj_est_training_features.json",
                                set_internal=True):
        output = Path(DATASET_DIR).joinpath(features_output)
        # if Path(output).exists():
        #     print("Found generated output file, skipping!")
        # else:
        print("Generating training file...")
        tqdm.pandas(desc="Generating features.")
        training_data[self.features] = training_data.progress_apply(
            lambda x: self.generate_features(x["content"], train=True), axis=1, result_type="expand")
        training_data.to_json(output)
        training_data.to_csv(Path(DATASET_DIR).joinpath("redorank_obj_est_training_features.csv"))
        if set_internal:
            self.train_file = features_output

    def predict_proba(self, resource):
        return self.classifier.predict_proba([self.generate_features(resource)])[0][1]

    def predict(self, resource):
        return self.classifier.predict([self.generate_features(resource)])

    def fit(self, data, train=False):
        self._generate_training_file(data)
        time.sleep(1)
        filepath = Path(DATASET_DIR).joinpath(self.train_file)
        train_data = pd.read_json(filepath, orient="columns")
        X = train_data[self.features]
        y = train_data.label

        if train:
            print("Optimizing internal prediction model...")
            best_clfs = []
            skf = StratifiedKFold(n_splits=5)
            splits = tqdm(skf.split(X, y))
            split = 1
            for train_index, test_index in splits:
                splits.set_description(desc=f"Running Split #{split}")
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                pipe = Pipeline(steps=[('random_forest', self.classifier)])
                param_grid = {
                    'random_forest__max_depth': [2, 4, 8, 16, 32, 64, 128],
                    'random_forest__max_leaf_nodes': [2, 4, 8, 16, 32, 64, 128],
                    'random_forest__min_samples_leaf': [2, 4, 8, 16, 32, 64, 128],
                    'random_forest__min_samples_split': [2, 4, 8, 16, 32, 64, 128],
                }

                search = GridSearchCV(pipe, param_grid, n_jobs=-1)
                search.fit(X_train, y_train)
                results = {"model": search.best_estimator_, "params": search.best_params_, "score": search.best_score_}

                print(results)

                best_clfs.append(results)
                split += 1

            score_sorted = sorted(best_clfs, key=lambda k: k["score"], reverse=True)
            best = score_sorted[0]
            print(best["params"])
            self.classifier = best["model"]
        else:
            print("Training internal prediction model.")
            self.classifier.fit(X, y)
