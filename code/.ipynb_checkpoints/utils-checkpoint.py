from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

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
    

