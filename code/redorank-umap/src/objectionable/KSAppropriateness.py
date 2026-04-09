import enchant
import pandas as pd
import nltk
import re
import string
import time
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


DATASET_DIR = str(Path(__file__).resolve().parent.parent.parent.parent.joinpath("datasets", "objectionable"))


class AppropriatenessEstimation(object):
    # Initializer / Instance Attributes
    def __init__(self, clf="random_forest"):
        self.train_file = "maro_ks_app_train_data.json"
        sexually_explicit_words = pd.read_csv(Path(DATASET_DIR).joinpath(
            "korsce", "appropriateness", "sexually_explicit_words.csv"), names=["words"])
        self.sexually_explicit_words = sexually_explicit_words.words.to_list()
        hate_speech_words = pd.read_csv(Path(DATASET_DIR).joinpath(
            "korsce", "appropriateness", "hate_speech_words.csv"), names=["words"])
        self.hate_speech_words = hate_speech_words.words.to_list()
        self.features = ["inap_count_doc", "inap_prop_doc", "misp_count_doc", "misp_prop_doc", "hate_count_doc",
                         "hate_prop_doc", "inap_count_meta", "inap_prop_meta", "misp_count_meta", "misp_prop_meta",
                         "hate_count_meta",  "hate_prop_meta", "inap_count_anc", "inap_prop_anc", "misp_count_anc",
                         "misp_prop_anc", "hate_count_anc", "hate_prop_anc"]
        self.d = enchant.Dict("en_US")
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.stops = stopwords.words('english')
        if clf == "random_forest":
            self.classifier = RandomForestClassifier(
                criterion='gini', max_depth=8, max_leaf_nodes=32, min_samples_leaf=32, min_samples_split=22)
        elif clf == "mlp":
            self.classifier = MLPClassifier(activation='relu', alpha=0.05, hidden_layer_sizes=(50, 100, 50),
                                            learning_rate='constant', solver='adam')


    @staticmethod
    def _get_inappropriate_words(doc_words, lexicon):
        for x in doc_words:
            for y in lexicon:
                if str(x).startswith(str(y)) or str(x) == str(y):
                    if str(x) in doc_words:
                        doc_words[doc_words.index(x)] = y

        return [x for x in doc_words if x in lexicon]

    def _get_misspelled_words(self, words):
        misspelled = [w if self.d.check(w) is False else "" for w in words]
        misspelled = list(set([w for w in misspelled if w]))
        for x in misspelled:
            for y in set(list(self.sexually_explicit_words + self.hate_speech_words)):
                if str(x).startswith(str(y)) or str(x).endswith(str(y)) or x == y:
                    if x in misspelled:
                        misspelled[misspelled.index(x)] = y

        return misspelled, [x for x in misspelled if x
                            in set(list(self.sexually_explicit_words + self.hate_speech_words))]

    @staticmethod
    def _get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    def _tokenize_and_clean(self, content):
        tokens = [nltk.word_tokenize(content.lower().replace("–", " "))]
        flattened_tokens = [s for k in tokens for s in k]
        flattened_tokens = [re.sub(r'[^\x00-\x7F]+', '', s) for s in flattened_tokens]
        flattened_tokens = [s for s in flattened_tokens if s]
        stemmed_tokens = [self.lemmatizer.lemmatize(s, self._get_wordnet_pos(s)).strip() for s in flattened_tokens]
        cleaned_tokens = [''.join(c for c in s if c not in string.punctuation) for s in stemmed_tokens]
        cleaned_tokens = [w for w in cleaned_tokens if w and len(w) in range(2, 15)]
        return cleaned_tokens
    
    def unique_inappropriate_words_count(self, doc_words):
        inap_words = self._get_inappropriate_words(list(set(doc_words)), self.sexually_explicit_words)

        return len(inap_words)
    
    def inappropriate_words_prop(self, doc_words):
        inap_words = self._get_inappropriate_words(doc_words, self.sexually_explicit_words)

        if len(doc_words) == 0:
            score = 0.0
        else: 
            score = len(inap_words) / len(doc_words)
        return score
    
    def unique_hatebased_words_count(self, doc_words):
        inap_words = self._get_inappropriate_words(list(set(doc_words)), self.hate_speech_words)

        return len(inap_words)

    def hatebased_words_prop(self, doc_words):
        inap_words = self._get_inappropriate_words(doc_words, self.hate_speech_words)

        if len(doc_words) == 0:
            score = 0.0
        else: 
            score = len(inap_words) / len(doc_words)
        return score
    
    def unique_misspelled_words_count(self, words):
        _, inap_misspelled = self._get_misspelled_words(words)

        return len(inap_misspelled)
    
    def misspelled_words_prop(self, words):
        misspelled, inap_misspelled = self._get_misspelled_words(words)

        if len(misspelled) == 0:
            score = 0.0
        else: 
            score = len(inap_misspelled) / len(misspelled)

        return score
    
    def feature_names(self):
        return self.features
    
    def generate_filtering_features(self, resource, metatag, anchortag):
        cleaned_doc = self._tokenize_and_clean(resource)
        cleaned_meta = self._tokenize_and_clean(metatag)
        cleaned_anchor = self._tokenize_and_clean(anchortag)

        inap_count_doc = self.unique_inappropriate_words_count(cleaned_doc)
        inap_prop_doc = self.inappropriate_words_prop(cleaned_doc)
        misp_count_doc = self.unique_misspelled_words_count(cleaned_doc)
        misp_prop_doc = self.misspelled_words_prop(cleaned_doc)
        hate_count_doc = self.unique_hatebased_words_count(cleaned_doc)
        hate_prop_doc = self.hatebased_words_prop(cleaned_doc)

        inap_count_meta = self.unique_inappropriate_words_count(cleaned_meta)
        inap_prop_meta = self.inappropriate_words_prop(cleaned_meta)
        misp_count_meta = self.unique_misspelled_words_count(cleaned_meta)
        misp_prop_meta = self.misspelled_words_prop(cleaned_meta)
        hate_count_meta = self.unique_hatebased_words_count(cleaned_meta)
        hate_prop_meta = self.hatebased_words_prop(cleaned_meta)

        inap_count_anc = self.unique_inappropriate_words_count(cleaned_anchor)
        inap_prop_anc = self.inappropriate_words_prop(cleaned_anchor)
        misp_count_anc = self.unique_misspelled_words_count(cleaned_anchor)
        misp_prop_anc = self.misspelled_words_prop(cleaned_anchor)
        hate_count_anc = self.unique_hatebased_words_count(cleaned_anchor)
        hate_prop_anc = self.hatebased_words_prop(cleaned_anchor)

        return [inap_count_doc, inap_prop_doc, misp_count_doc, misp_prop_doc, hate_count_doc,
                hate_prop_doc, inap_count_meta, inap_prop_meta, misp_count_meta, misp_prop_meta,
                hate_count_meta, hate_prop_meta, inap_count_anc, inap_prop_anc, misp_count_anc,
                misp_prop_anc, hate_count_anc, hate_prop_anc]

    def _generate_training_file(self, training_data, features_output="maro_ks_app_train_data.json",
                                set_internal=True):
        output = Path(DATASET_DIR).joinpath("korsce", "appropriateness", features_output)
        # if Path(output).exists():
        #     print("Found generated output file, skipping!")
        # else:
        print("Generating training file...")
        tqdm.pandas(desc="Generating features.")
        training_data[self.features] = training_data.progress_apply(
            lambda x: self.generate_filtering_features(x["content"], "", ""), axis=1, result_type="expand")
        training_data.to_json(output)
        if set_internal:
            self.train_file = features_output

    def predict(self, resource, metatag, anchortag):
        probs = self.classifier.predict_proba([self.generate_filtering_features(resource, metatag, anchortag)])[0][0]
        predictions = self.classifier.predict([self.generate_filtering_features(resource, metatag, anchortag)])
        return probs, predictions

    def fit(self, train_data):
        self._generate_training_file(train_data)
        time.sleep(1)
        filepath = Path(DATASET_DIR).joinpath("korsce", "appropriateness", self.train_file)
        safe_unsafe_features = pd.read_json(filepath, orient="columns")
        feat = safe_unsafe_features[self.features]
        target = safe_unsafe_features.label

        self.classifier.fit(feat, target)
