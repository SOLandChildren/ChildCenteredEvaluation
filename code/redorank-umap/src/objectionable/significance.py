from pathlib import Path
from sklearn.metrics import confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def bonferroni_correction(p_values, data):
    tests = len(p_values)
    reject, corrected, _, alphaBonf = multipletests(p_values, method='bonferroni')
    for test in range(tests):
        print("MODEL:\t\t", data[test], "\t\tRESULT:\t\t", reject[test], "\t\t:", alphaBonf)


def build_contingency_table(frame):
    yesyes = frame.apply(lambda x: 1 if x['clf1'] == True and x['clf2'] == True else 0, axis=1).value_counts()[1]
    yesno = frame.apply(lambda x: 1 if x['clf1'] == True and x['clf2'] == False else 0, axis=1).value_counts()[1]
    noyes = frame.apply(lambda x: 1 if x['clf1'] == False and x['clf2'] == True else 0, axis=1).value_counts()[1]
    nono = frame.apply(lambda x: 1 if x['clf1'] == False and x['clf2'] == False else 0, axis=1).value_counts()[1]

    table = [[yesyes, yesno],
             [noyes, nono]]

    return table


def build_frame(frame1, frame2, ground_truth_column, prediction_column):
    """Compares the `edu` and `prediction` columns"""
    df = pd.DataFrame()
    df['clf1'] = frame1.apply(lambda x: x[ground_truth_column] == x[prediction_column], axis=1)
    df['clf2'] = frame2.apply(lambda x: x[ground_truth_column] == x[prediction_column], axis=1)
    return df


def calculate_mcnemars(table):
    m = mcnemar(table, exact=True)
    return m.pvalue


def assign_prediction_type(item, truth_column, prediction_column):
    if item[truth_column] == 0 and item[prediction_column] == 1:
        return "FP"
    elif item[truth_column] == 1 and item[prediction_column] == 0:
        return "FN"
    elif item[truth_column] == 1 and item[prediction_column] == 1:
        return "TP"
    elif item[truth_column] == 0 and item[prediction_column] == 0:
        return "TN"


def prediction_analysis(frame, truth_column, prediction_column, prob_column, model_name):
    dataset_dir = str(Path(__file__).resolve().parent.parent.parent.joinpath("datasets", "objectionable"))
    sub_frame = frame.iloc[:500].copy()
    sub_frame["prediction_type"] = sub_frame.apply(assign_prediction_type, args=(truth_column, prediction_column), axis=1)
    sub_frame.sort_values(by=[prob_column], inplace=True, ascending=False)
    sub_frame.reset_index(inplace=True, drop=True)

    if model_name == "Objectionability Estimator":
        # plt.figure(figsize=(20, 20))
        exploratory_fn = frame.loc[(frame[truth_column] == 1) & (frame[prediction_column] == 0)]
        exploratory_fp = frame.loc[(frame[truth_column] == 0) & (frame[prediction_column] == 1)]
        # heat_plot = sns.heatmap(exploratory_fn.corr())
        # heat_fig = heat_plot.get_figure()
        # heat_fig.savefig(Path(dataset_dir).joinpath("images", f"{model_name}_correlation_heat_map.png"))
        exploratory_fn.to_csv(
            Path(dataset_dir).joinpath("images", f"exploratory_fn_{model_name.replace(' ', '_')}.csv"))
        exploratory_fp.to_csv(
            Path(dataset_dir).joinpath("images", f"exploratory_fp_{model_name.replace(' ', '_')}.csv"))

    plt.figure(figsize=(6, 4))
    plot = sns.scatterplot(x=sub_frame.index, y=sub_frame[prob_column], hue=sub_frame["prediction_type"])
    plot.set(ylim=(0, 1))

    plot.figure.show()
    fig = plot.get_figure()
    fig.savefig(Path(dataset_dir).joinpath("images", f"{model_name}_prediction_plot.png"))


def fp_fn_rate_analysis(frame, truth_column, prediction_column, model_name):
    print(f"Confusion Matrix: {model_name}\n")
    tn, fp, fn, tp = confusion_matrix(frame[truth_column], frame[prediction_column]).ravel()
    print(f"TP:\t{tp}\nTN:\t{tn}\nFP:\t{fp}\nFN:\t{fn}")

    fpr_denom = fp + tn
    if fpr_denom == 0:
        fpr = 0.0
    else:
        fpr = fp / fpr_denom
    fnr = fn / (tp + fn)

    print(f"FPR:\t\t{fpr}")
    print(f"FNR:\t\t{fnr}")


def run():
    dataset_dir = str(Path(__file__).resolve().parent.parent.parent.joinpath("datasets", "objectionable"))
    prediction_data = pd.read_csv(Path(dataset_dir).joinpath("validation_data_with_predictions.csv"))

    # Set up frames for comparison
    nb_frame = prediction_data[["label", "nb_prediction", "nb_probs"]].copy()
    nb_frame.rename(columns={"nb_prediction": "prediction"}, inplace=True)
    prediction_analysis(nb_frame, "label", "prediction", "nb_probs", "Naive Bayes")

    awessome_frame = prediction_data[["label", "awessome_prediction"]].copy()
    awessome_frame.rename(columns={"awessome_prediction": "prediction"}, inplace=True)

    ks_app_frame = prediction_data[["label", "ks_app_prediction"]].copy()
    ks_app_frame.rename(columns={"ks_app_prediction": "prediction"}, inplace=True)

    bert4tc_frame = prediction_data[["label", "bert_4_tc_prediction"]].copy()
    bert4tc_frame.rename(columns={"bert_4_tc_prediction": "prediction"}, inplace=True)

    oe_frame = prediction_data.copy()
    oe_frame.rename(columns={"ap_prediction": "prediction"}, inplace=True)
    prediction_analysis(oe_frame, "label", "prediction", "ap_probs", "Objectionability Estimator")

    # Build comparison frames
    oe_vs_nb = build_frame(oe_frame, nb_frame, "label", "prediction")
    oe_vs_bert4tc = build_frame(oe_frame, bert4tc_frame, "label", "prediction")
    oe_vs_ks_app = build_frame(oe_frame, ks_app_frame, "label", "prediction")
    oe_vs_awessome = build_frame(oe_frame, awessome_frame, "label", "prediction")

    # Build contingency tables
    baselines = []
    baselines.append({"comparison": "ObjectionablityEstimator vs Naive-Bayes",
                      "table": build_contingency_table(oe_vs_nb)})
    baselines.append({"comparison": "ObjectionablityEstimator vs BERT4TC",
                      "table": build_contingency_table(oe_vs_bert4tc)})
    baselines.append({"comparison": "ObjectionablityEstimator vs AWESSOME",
                      "table": build_contingency_table(oe_vs_awessome)})
    baselines.append({"comparison": "ObjectionablityEstimator vs KSAppropriateness",
                      "table": build_contingency_table(oe_vs_ks_app)})

    # Calculate McNemar's p
    bl_comparisons = []
    bl_p_values = []
    for table in baselines:
        bl_comparisons.append(table["comparison"])
        bl_p_values.append(calculate_mcnemars(table["table"]))

    for item in zip(bl_comparisons, bl_p_values):
        print(item)

    # Bonferroni Correction
    bonferroni_correction(bl_p_values, bl_comparisons)

    print("\n\nObjectionablityEstimator")
    fp_fn_rate_analysis(oe_frame, "label", "prediction", "Objectionablility Estimator")
    print("-" * 75)
    print("Naive-Bayes")
    fp_fn_rate_analysis(nb_frame, "label", "prediction", "Naive Bayes")
    print("-" * 75)
    print("KSAppropriateness")
    fp_fn_rate_analysis(ks_app_frame, "label", "prediction", "KSAppropriateness")
    print("-" * 75)
    print("AWESSOME")
    fp_fn_rate_analysis(awessome_frame, "label", "prediction", "AWESSOME")
    print("-" * 75)
    print("BERT4TC")
    fp_fn_rate_analysis(bert4tc_frame, "label", "prediction", "BERT4TC")
    print("-" * 75)

    print("Dataset Distribution")
    print(prediction_data["label"].value_counts())
    print("-" * 75)


if __name__ == "__main__":
    run()