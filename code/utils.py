from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
import pandas as pd

def get_ranked_lists(path):
    bm25 = pd.read_csv(path+"BM25.res", sep=" ", header=None, names=["qid", "q0", "docno", "rank", "score", "pyterrier"])
    tfidf = pd.read_csv(path+"TF-IDF.res", sep=" ", header=None, names=["qid", "q0", "docno", "rank", "score", "pyterrier"])
    dlm = pd.read_csv(path+"DirichletLM.res", sep=" ", header=None, names=["qid", "q0", "docno", "rank", "score", "pyterrier"])
    monot5 = pd.read_csv(path+"MonoT5-base.res", sep=" ", header=None, names=["qid", "q0", "docno", "rank", "score", "pyterrier"])
    vicuna = pd.read_csv(path+"RankVicuna.res", sep=" ", header=None, names=["qid", "q0", "docno", "rank", "score", "pyterrier"])
    zephyr = pd.read_csv(path+"RankZephyr.res", sep=" ", header=None, names=["qid", "q0", "docno", "rank", "score", "pyterrier"])

    return [tfidf, bm25, dlm, monot5, vicuna, zephyr]


def RBP(topics, qrels, retriever_system, k=None, phi=0.8, perquery=False):
    if isinstance(k, int):
        run = retriever_system(topics)
        RBP_scores = []
        
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
                query_score += (phi**(rank-1)) * rel
            query_score = query_score * (1-phi)
            RBP_scores.append(query_score)
    elif k==None:
        run = retriever_system(topics)
        RBP_scores = []
        
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
    
def chRBP(topics, topical_qrels, readability_qrels, harm_qrels, retriever_system, k=None, phi=0.8, perquery=False):
    pass
    # if isinstance(k, int):
    #     run = retriever_system(topics)
    #     uhRBP_scores = []
        
    #     for qid in topics["qid"].unique():
    #         query_score = 0
    #         docs = run.loc[run["qid"]==qid][:k] # the top-k documents retrieved for a query
    #         docid_col_name = [col for col in topical_qrels.columns if col.startswith("doc")][0] # the name of the column containing the docid
    #         for _, row in docs.iterrows():
    #             rank = row["rank"] + 1
    #             docid = row[docid_col_name]
                
    #             topical_qrel_row = topical_qrels.loc[(topical_qrels["qid"]==qid) & (topical_qrels[docid_col_name]==docid)]
    #             if len(topical_qrel_row) == 0:
    #                 topical_rel = 0
    #             elif len(topical_qrel_row) == 1:
    #                 topical_rel = list(topical_qrel_row["relevance"])[0]
    #             else:
    #                 raise ValueError("query-document pair has more than one topical relevance label.")
                
    #             harm_qrel_row = harm_qrels.loc[(harm_qrels["qid"]==qid) & (harm_qrels[docid_col_name]==docid)]
    #             if len(harm_qrel_row) == 0:
    #                 harm_rel = 0
    #             elif len(harm_qrel_row) == 1:
    #                 harm_rel = list(harm_qrel_row["relevance"])[0]
    #             else:
    #                 raise ValueError("query-document pair has more than one harm relevance label.")
                
    #             read_qrel_row = readability_qrels.loc[(readability_qrels["qid"]==qid) & (readability_qrels[docid_col_name]==docid)]
    #             if len(read_qrel_row) == 0:
    #                 read_rel = 0
    #             elif len(read_qrel_row) == 1:
    #                 read_rel = list(read_qrel_row["relevance"])[0]
    #             else:
    #                 raise ValueError("query-document pair has more than one readability relevance label.")
                
    #             query_score += (phi**(rank-1)) * topical_rel * read_rel * harm_rel
    #         query_score = query_score * (1-phi)
    #         uhRBP_scores.append(query_score)
    # elif k==None:
    #     run = retriever_system(topics)
    #     uhRBP_scores = []
        
    #     for qid in topics["qid"].unique():
    #         query_score = 0
    #         docs = run.loc[run["qid"]==qid] # all documents retrieved for a query
    #         docid_col_name = [col for col in topical_qrels.columns if col.startswith("doc")][0] # the name of the column containing the docid
    #         for _, row in docs.iterrows():
    #             rank = row["rank"] + 1
    #             docid = row[docid_col_name]
    #             topical_qrel_row = topical_qrels.loc[(topical_qrels["qid"]==qid) & (topical_qrels[docid_col_name]==docid)]
    #             if len(topical_qrel_row) == 0:
    #                 topical_rel = 0
    #             elif len(topical_qrel_row) == 1:
    #                 topical_rel = list(topical_qrel_row["relevance"])[0]
    #             else:
    #                 raise ValueError("query-document pair has more than one topical relevance label.")
                
    #             harm_qrel_row = harm_qrels.loc[(harm_qrels["qid"]==qid) & (harm_qrels[docid_col_name]==docid)]
    #             if len(harm_qrel_row) == 0:
    #                 harm_rel = 0
    #             elif len(harm_qrel_row) == 1:
    #                 harm_rel = list(harm_qrel_row["relevance"])[0]
    #             else:
    #                 raise ValueError("query-document pair has more than one harm relevance label.")
                
    #             read_qrel_row = readability_qrels.loc[(readability_qrels["qid"]==qid) & (readability_qrels[docid_col_name]==docid)]
    #             if len(read_qrel_row) == 0:
    #                 read_rel = 0
    #             elif len(read_qrel_row) == 1:
    #                 read_rel = list(read_qrel_row["relevance"])[0]
    #             else:
    #                 raise ValueError("query-document pair has more than one readability relevance label.")
                
    #             query_score += (phi**(rank-1)) * topical_rel * read_rel * harm_rel
    #         query_score = query_score * (1-phi)
    #         uhRBP_scores.append(query_score)
    # else:
    #     raise ValueError("cutoff value not valid.")
    # if perquery == True:
    #     return uhRBP_scores
    # else:
    #     assert len(uhRBP_scores) == len(topics)
    #     return sum(uhRBP_scores)/len(uhRBP_scores)
    
def get_graded_readability_milton(readability_val, th):
    if readability_val == th:
        return 1
    elif (readability_val < (th + 4)) and (readability_val > th):
        return (np.cos(0.79*readability_val - (th - (0.21*th)))+1)/2
    elif (readability_val < th) and (readability_val > (th-6)):
        return (np.cos((0.5236 * readability_val) - (0.5236 * th))+1)/2
    else:
        return 0
        
    # return (1/2)-(np.arctan(readability_val-th)/np.pi)

def get_graded_readability_rel_milton(readability_scores, threshold):
    labels = []
    for score in readability_scores:
        if score < 0:
            labels.append(None)
        else:
            labels.append(get_graded_readability_milton(int(score), th=threshold))
    return labels

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
    pipe = pipeline("text-classification", model="GroNLP/mdebertav3-subjectivity-english", top_k=2) # to directly get the prediction
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
    

