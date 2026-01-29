from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
import pandas as pd
import pyterrier as pt

def get_ranked_lists(path, dataset_name):
    bm25 = pd.read_csv(path+"BM25.res", sep=" ", header=None, names=["qid", "Q0", "docno", "rank", "score", "system"]).sort_values(by=["qid", "rank"], axis=0)
    tfidf = pd.read_csv(path+"TF-IDF.res", sep=" ", header=None, names=["qid", "Q0", "docno", "rank", "score", "system"]).sort_values(by=["qid", "rank"], axis=0)
    dlm = pd.read_csv(path+"DirichletLM.res", sep=" ", header=None, names=["qid", "Q0", "docno", "rank", "score", "system"]).sort_values(by=["qid", "rank"], axis=0)
    monot5 = pd.read_csv(path+"MonoT5-base.res", sep=" ", header=None, names=["qid", "Q0", "docno", "rank", "score", "system"]).sort_values(by=["qid", "rank"], axis=0)
    vicuna = pd.read_csv(path+"RankVicuna.res", sep=" ", header=None, names=["qid", "Q0", "docno", "rank", "score", "system"]).sort_values(by=["qid", "rank"], axis=0)
    zephyr = pd.read_csv(path+"RankZephyr.res", sep=" ", header=None, names=["qid", "Q0", "docno", "rank", "score", "system"]).sort_values(by=["qid", "rank"], axis=0)
    korsce = pd.read_csv(path+"korsce.txt", sep=" ", header=None, names=["qid", "Q0", "docno", "rank", "score", "system"]).sort_values(by=["qid", "rank"], axis=0)
    redorank = pd.read_csv(path+"redorank.txt", sep=" ", header=None, names=["qid", "Q0", "docno", "rank", "score", "system"]).sort_values(by=["qid", "rank"], axis=0)
    
    if dataset_name == "kidfriend":
        google = pd.read_csv("results/kid-friend/google.txt", sep=" ", header=None, names=["qid", "Q0", "docno", "rank", "score", "system"]).sort_values(by=["qid", "rank"], axis=0)
        google_korsce = pd.read_csv("results/kid-friend/google_korsce.txt", sep=" ", header=None, names=["qid", "Q0", "docno", "rank", "score", "system"]).sort_values(by=["qid", "rank"], axis=0)
        google_redorank = pd.read_csv("results/kid-friend/google_redorank.txt", sep=" ", header=None, names=["qid", "Q0", "docno", "rank", "score", "system"]).sort_values(by=["qid", "rank"], axis=0)
        
        bing = pd.read_csv("results/kid-friend/bing.txt", sep=" ", header=None, names=["qid", "Q0", "docno", "rank", "score", "system"]).sort_values(by=["qid", "rank"], axis=0)
        bing_korsce = pd.read_csv("results/kid-friend/bing_korsce.txt", sep=" ", header=None, names=["qid", "Q0", "docno", "rank", "score", "system"]).sort_values(by=["qid", "rank"], axis=0)
        bing_redorank = pd.read_csv("results/kid-friend/bing_redorank.txt", sep=" ", header=None, names=["qid", "Q0", "docno", "rank", "score", "system"]).sort_values(by=["qid", "rank"], axis=0)
        
        return [bm25, tfidf, dlm, monot5, vicuna, zephyr, korsce, redorank, google, google_korsce, google_redorank, bing, bing_korsce, bing_redorank]
    
    elif dataset_name == "requik":    
        return [bm25, tfidf, dlm, monot5, vicuna, zephyr, korsce, redorank]



def RBP(topics, qrels, retriever_system, k=None, phi=0.8, perquery=False):
    if isinstance(retriever_system, pt.terrier.retriever.Retriever):
        run = retriever_system(topics)
    elif isinstance(retriever_system, pd.DataFrame):
        run = retriever_system

    run["qid"] = run["qid"].astype(str)
    # run = run.sort_values(by=["qid", "rank"], axis=0)
    RBP_scores = []
    if isinstance(k, int):
        for qid in topics["qid"].unique():
            # print(type(qid), run["qid"].dtype)
            query_score = 0
            docs = run.loc[run["qid"]==qid][:k] # the top-k documents retrieved for a query
            # print(docs)
            # break
            docid_col_name = [col for col in qrels.columns if col.startswith("doc")][0] # the name of the column containing the docid
            for _, row in docs.iterrows():
                rank = row["rank"] + 1
                docid = row[docid_col_name]
                qrel_row = qrels.loc[(qrels["qid"]==qid) & (qrels[docid_col_name]==docid)]
                if len(qrel_row) == 0:
                    rel = 0
                elif len(qrel_row) == 1:
                    rel = list(qrel_row["relevance"])[0]
                else:
                    raise ValueError("query-document pair has more than one relevance label.")
                query_score += (phi**(rank-1)) * rel
            query_score = query_score * (1-phi)
            RBP_scores.append(query_score)
    elif k==None:
        for qid in topics["qid"].unique():
            query_score = 0
            docs = run.loc[run["qid"]==qid] # all documents retrieved for a query
            docid_col_name = [col for col in qrels.columns if col.startswith("doc")][0] # the name of the column containing the docid
            for _, row in docs.iterrows():
                rank = row["rank"] + 1
                docid = row[docid_col_name]
                qrel_row = qrels.loc[(qrels["qid"]==qid) & (qrels[docid_col_name]==docid)]
                if len(qrel_row) == 0:
                    rel = 0
                elif len(qrel_row) == 1:
                    rel = list(qrel_row["relevance"])[0]
                else:
                    raise ValueError("query-document pair has more than one relevance label.")
                query_score += (phi**(rank-1)) * rel
            query_score = query_score * (1-phi)
            RBP_scores.append(query_score)
    else:
        raise ValueError("cutoff value not valid.")
    if perquery == True:
        return RBP_scores
    else:
        assert len(RBP_scores) == len(topics)
        return sum(RBP_scores)/len(RBP_scores)
    
def cRBP(topics, corpus, qrels, retriever_system, upper_threshold, lower_threshold, k=None, phi=0.8, perquery=False):
    if isinstance(retriever_system, pt.terrier.retriever.Retriever):
        run = retriever_system(topics)
        run["qid"] = run["qid"].astype(str)
    elif isinstance(retriever_system, pd.DataFrame):
        run = retriever_system
        
    run["qid"] = run["qid"].astype(str)
    # run = run.sort_values(by=["qid", "rank"], axis=0)
    RBP_scores = []
    if isinstance(k, int):
        
        for qid in topics["qid"].unique():
            query_score = 0
            docs = run.loc[run["qid"]==qid][:k] # the top-k documents retrieved for a query
            docid_col_name = [col for col in qrels.columns if col.startswith("doc")][0] # the name of the column containing the docid
            for _, row in docs.iterrows():
                rank = row["rank"] + 1
                docid = row[docid_col_name]
                qrel_row = qrels.loc[(qrels["qid"]==qid) & (qrels[docid_col_name]==docid)]
                if len(qrel_row) == 0:
                    rel = 0
                elif len(qrel_row) == 1:
                    rel = list(qrel_row["relevance"])[0]
                else:
                    raise ValueError("query-document pair has more than one relevance label.")
                retr_doc = corpus.loc[corpus[docid_col_name]==docid]
                read =  get_comprehension_score(list(retr_doc["readability"])[0], th_high = upper_threshold, th_low=lower_threshold)
                obj = list(retr_doc["obj_prob"])[0]
                edu = list(retr_doc["edu_val"])[0]/5
                query_score += (phi**(rank-1)) * rel * read * obj * edu
            query_score = query_score * (1-phi)
            RBP_scores.append(query_score)
    elif k==None:
        for qid in topics["qid"].unique():
            query_score = 0
            docs = run.loc[run["qid"]==qid] # all documents retrieved for a query
            docid_col_name = [col for col in qrels.columns if col.startswith("doc")][0] # the name of the column containing the docid
            for _, row in docs.iterrows():
                rank = row["rank"] + 1
                docid = row[docid_col_name]
                qrel_row = qrels.loc[(qrels["qid"]==qid) & (qrels[docid_col_name]==docid)]
                if len(qrel_row) == 0:
                    rel = 0
                elif len(qrel_row) == 1:
                    rel = list(qrel_row["relevance"])[0]
                else:
                    raise ValueError("query-document pair has more than one relevance label.")
                retr_doc = corpus.loc[corpus[docid_col_name]==docid]
                read =  get_comprehension_score(list(retr_doc["readability"])[0], th_high = upper_threshold, th_low=lower_threshold)
                obj = list(retr_doc["obj_prob"])[0]
                edu = list(retr_doc["edu_val"])[0]/5
                query_score += (phi**(rank-1)) * rel * read * obj * edu
            query_score = query_score * (1-phi)
            RBP_scores.append(query_score)
    else:
        raise ValueError("cutoff value not valid.")
    if perquery == True:
        return RBP_scores
    else:
        assert len(RBP_scores) == len(topics)
        return sum(RBP_scores)/len(RBP_scores)
    
def get_comprehension_score(readability_val, th_high, th_low):
    if (th_low <= readability_val) and (readability_val <= th_high): # readability within expected range
        return 1
    elif (th_high < readability_val) and (readability_val < (th_high + 4)): # readability higher than upper threshold
        return (np.cos(0.79*readability_val - (th_high - (0.21*th_high)))+1)/2
    elif ((th_low - 6) < readability_val) and (readability_val < th_low): # readability less than lower threshold
        return (np.cos((0.5236 * readability_val) - (0.5236 * th_low))+1)/2
    else:
        return 0
        
    # return (1/2)-(np.arctan(readability_val-th)/np.pi)

# def get_graded_readability_rel_milton(readability_scores, upper_threshold, lower_threshold):
#     labels = []
#     for score in readability_scores:
#         if score < 0:
#             labels.append(None)
#         else:
#             labels.append(get_graded_readability_milton(int(score), th_high=upper_threshold, th_low=lower_threshold))
#     return labels

def get_edu_value(text):
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")
    model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")
    # text = "This is a test sentence."
    inputs = tokenizer(text, return_tensors="pt", padding="longest", truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits.squeeze(-1).float().detach().numpy()
    score = logits.item()
    result = {
        "text": text,
        "score": score,
        "int_score": int(round(max(0, min(score, 5)))),
    }
    return result["int_score"]

def get_obj_likelihood(texts):
    pipe = pipeline("text-classification", model="GroNLP/mdebertav3-subjectivity-english", top_k=2)
    # label_map = {
    #     "LABEL_0": "OBJ",
    #     "LABEL_1": "SUBJ"
    # }
    result_lists = pipe(texts)
    obj_scores = []
    for result in result_lists:
        for pred in result:
            if pred["label"] == "LABEL_0":
                obj_scores.append(pred["score"])
                break
    return obj_scores

def pred_obj(texts):
    label_map = {
        "LABEL_0": "OBJ",
        "LABEL_1": "SUBJ"
    }
    pipe = pipeline("text-classification", model="GroNLP/mdebertav3-subjectivity-english") # to directly get the prediction
    result_list = pipe(texts)
    return [int(label_map[result["label"]]=="OBJ") for result in result_list]
    

