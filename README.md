# It is relevant, but is it useful? A Reflection on Human-Centred Evaluation in Children Information Retrieval

# Abstract
The traditional Information Retrieval (IR) evaluation framework---anchored in topical relevance and relevance‑based metrics---reflects a system-centred perspective. Yet for specific user groups, relevance alone is insufficient---benchmarking that relies exclusively on conventional metrics overlooks qualities intrinsic to the users IR approaches are meant to serve. Here, we draw attention to Children IR and examine the value of extending traditional evaluation with a human-centred perspective that accounts for how children interpret and evaluate information to more authentically capture performance and better reflect how well an approach truly meets children's needs. Our empirical exploration using a child‑focused dataset, multiple ranking strategies, and traditional and extended frameworks reveals not only the limitations of relevance-based assessments but also the advantages of employing frameworks that are tailored to reflect the needs of child users, paving the way for more inclusive and effective evaluation frameworks.

# Data
1. The kid-friend dataset can be downloaded from the public [Zenodo repository](https://zenodo.org/records/18076554). Unzip the dataset in the ``data\`` directory and rename the dataset directory to kid-friend-en.
**Note: We use the english dataset, i.e., kid-friend-en.**
2. To compute readability, clone the [Spache-Allen GitHub repo](https://github.com/BSU-CAST/ecir22-readability/tree/main) into the ``code\`` directory.

# Experimental code
Run the following python notebooks in order:
1. ``kid_friend_preprocessing.ipynb`` to convert the data files into suitable format and compute judgment labels for readability, objectivity, and educational value.
2. ``indexing_retrieval.ipynb`` to index the kid-friend corpus and implement the rankers considered in the study.
3. To implement the REdORank, run the following commands on a terminal opened from within the ``code`` directory:  
``python redorank-umap/src/experiment.py --model kid-friend_bm25 -l --store_predictions`` to re-rank resources retrieved by BM25  
``python redorank-umap/src/experiment.py --model kid-friend_google -l --store_predictions`` to re-rank Google results  
``python redorank-umap/src/experiment.py --model kid-friend_bing -l --store_predictions`` to re-rank Bing results
4. ``evaluation.ipynb`` to benchmark ranker performance using the $Traditional$, $Multiview$, and $Composite$ frameworks.

