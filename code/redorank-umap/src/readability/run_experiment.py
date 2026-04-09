import matplotlib
import formulas
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import textstat as ts
from pathlib import Path
from tqdm import tqdm


FIGURE_SIZE = (35, 75)


def scale_readability_levels(frame, formula):
    """
    Function to scale the arbitrary scores of various readability formulas to grade levels
    """
    vals = frame[formula].values
    scaled = []
    if formula == "lix":
        for val in vals:
            if val <= 9:
                scaled.append(1)
            elif 9 < val <= 14:
                scaled.append(2)
            elif 14 < val <= 19:
                scaled.append(3)
            elif 19 < val <= 23:
                scaled.append(4)
            elif 23 < val <= 27:
                scaled.append(5)
            elif 27 < val <= 31:
                scaled.append(6)
            elif 31 < val <= 35:
                scaled.append(7)
            elif 35 < val <= 39:
                scaled.append(8)
            elif 39 < val <= 43:
                scaled.append(9)
            elif 43 < val <= 47:
                scaled.append(10)
            elif 47 < val <= 51:
                scaled.append(11)
            elif 51 < val <= 55:
                scaled.append(12)
            elif val > 55:
                scaled.append(13)
    elif formula == "rix":
        for val in vals:
            if val <= 0.19:
                scaled.append(1)
            elif 0.2 <= val < 0.5:
                scaled.append(2)
            elif 0.5 <= val < 0.8:
                scaled.append(3)
            elif 0.8 <= val < 1.3:
                scaled.append(4)
            elif 1.3 <= val < 1.8:
                scaled.append(5)
            elif 1.8 <= val < 2.4:
                scaled.append(6)
            elif 2.4 <= val < 3:
                scaled.append(7)
            elif 3 <= val < 3.7:
                scaled.append(8)
            elif 3.7 <= val < 4.5:
                scaled.append(9)
            elif 4.5 <= val < 5.3:
                scaled.append(10)
            elif 5.3 <= val < 6.2:
                scaled.append(11)
            elif 6.2 <= val < 7.2:
                scaled.append(12)
            elif val >= 7.2:
                scaled.append(13)
    elif formula == "dale_chall":
        for val in vals:
            if val <= 3:
                scaled.append(0)
            elif 3 <= val < 3.5:
                scaled.append(1)
            elif 3.5 <= val < 4:
                scaled.append(2)
            elif 4 <= val < 4.5:
                scaled.append(3)
            elif 4.5 <= val < 5:
                scaled.append(4)
            elif 5 <= val < 5.5:
                scaled.append(5)
            elif 5.5 <= val < 6:
                scaled.append(6)
            elif 6 <= val < 6.5:
                scaled.append(7)
            elif 6.5 <= val < 7:
                scaled.append(8)
            elif 7 <= val < 7.5:
                scaled.append(9)
            elif 7.5 <= val < 8:
                scaled.append(10)
            elif 8 <= val < 8.5:
                scaled.append(11)
            elif 8.5 <= val < 9:
                scaled.append(12)
            elif val >= 9:
                scaled.append(13)

    frame[formula] = scaled
    return frame


def assess_readability(frame, purpose="thesis"):
    """
    Function that calculates the readability scores for a collection of text passages
    """
    tqdm.pandas(desc="Applying Flesch-Kincaid")
    frame["fkg"] = frame["text"].progress_apply(lambda x: ts.flesch_kincaid_grade(x))
    tqdm.pandas(desc="Applying Spache (original)")
    frame["spache_original"] = frame["text"].progress_apply(lambda x: formulas.spache_readability_formula(x))
    if purpose == "thesis":
        tqdm.pandas(desc="Applying Spache (AoA < 11)")
        frame["spache_aoa_11"] = frame["text"].progress_apply(lambda x: formulas.spache_readability_formula_aoa_11(x))
        tqdm.pandas(desc="Applying Spache (AoA Full)")
        frame["spache_aoa_full"] = frame["text"].progress_apply(lambda x: formulas.spache_readability_formula_aoa_full(x))
        tqdm.pandas(desc="Applying Spache (Sven + AoA < 11)")
        frame["spache_sven_aoa_11"] = frame["text"].progress_apply(
            lambda x: formulas.spache_readability_formula_sven_aoa_11(x))
    tqdm.pandas(desc="Applying Spache (Sven)")
    frame["spache_sven"] = frame["text"].progress_apply(lambda x: formulas.spache_readability_formula_sven(x))
    tqdm.pandas(desc="Applying Spache-Allen")
    frame["spache_allen"] = frame["text"].progress_apply(
        lambda x: formulas.spache_allen(x))
    tqdm.pandas(desc="Applying Coleman-Liau Index")
    frame["cli"] = frame["text"].progress_apply(lambda x: ts.coleman_liau_index(x))
    tqdm.pandas(desc="Applying Gunning-Fog Index")
    frame["gfog"] = frame["text"].progress_apply(lambda x: ts.gunning_fog(x))
    tqdm.pandas(desc="Applying Lix")
    frame["lix"] = frame["text"].progress_apply(lambda x: ts.lix(x))
    tqdm.pandas(desc="Applying Rix")
    frame["rix"] = frame["text"].progress_apply(lambda x: ts.rix(x))
    tqdm.pandas(desc="Applying New Dale-Chall Reading Index")
    frame["dale_chall"] = frame["text"].progress_apply(lambda x: ts.dale_chall_readability_score_v2(x))
    tqdm.pandas(desc="Applying Simple Measure of Gobbledygook")
    frame["smog"] = frame["text"].progress_apply(lambda x: ts.smog_index(x))
    frame = scale_readability_levels(frame, "lix")
    frame = scale_readability_levels(frame, "rix")
    frame = scale_readability_levels(frame, "dale_chall")
    
    return frame


def calculate_error_rate(frame, medium, purpose="thesis", save=True):
    """
    Function to calculate the Error Rate of various readability formulas.
    """
    # Drop all -1 estimations for spache's and dale-chall
    frame = frame[(frame["dale_chall"] != -1) & (frame["spache_original"] != -1) &
                  (frame["spache_allen"] != -1) & (frame["spache_sven"] != -1)].copy()
    if purpose == "thesis":
        reads = ["fkg", "spache_original", "spache_aoa_11", "spache_sven", "spache_aoa_full", "spache_sven_aoa_11",
                 "spache_allen", "cli", "gfog", "lix", "rix", "dale_chall", "smog"]
    else:
        reads = ["fkg", "spache_original", "spache_sven", "spache_allen", "cli", "gfog", "lix", "rix", "dale_chall",
                 "smog"]
    for formula in reads:
        tqdm.pandas(desc=f"Calculating ER for {formula}")
        frame[f"{formula}_er"] = frame.progress_apply(lambda x: error_rate(x["grade_numeric"], x[formula]), axis=1)

    if save:
        frame.to_csv(Path().cwd().parent.parent.joinpath("datasets", "readability",
                                                         f"{purpose}_{medium}_readability_results.csv"), index=False)
    
    return frame


def error_rate(ground_truth, formula_result):
    """
    Formula to determine the margin of error for a readability assessment formula.
    """
    if formula_result >= 12.0:
        formula_result = 13.0
    if formula_result < 0:
        formula_result = 0.0
    return abs(float(ground_truth) - formula_result)


def grade_fill(row):
    """
    Function to convert Reading A-Z levels to grade levels
    """
    grade_conversion = {
        "aa": "K",
        "A": "K",
        "B": "K",
        "C": "K",
        "D": "1",
        "E": "1",
        "F": "1",
        "G": "1",
        "H": "1",
        "I": "1",
        "J": "1",
        "K": "2",
        "L": "2",
        "M": "2",
        "N": "2",
        "O": "2",
        "P": "2",
        "Q": "3",
        "R": "3",
        "S": "3",
        "T": "3",
        "U": "4",
        "V": "4",
        "W": "4",
        "X": "5",
        "Y": "5",
        "Z": "5",
        "Z1": "5+",
        "Z2": "5+"
    }

    if isinstance(row["Grade"], float):
        return grade_conversion[row["Level"]]


def load_books():
    """
    Function to load the datasets of books to be analyzed for readability levels.
    """
    books_raz_path = str(Path(__file__).resolve().parent.parent.parent.joinpath(
        "datasets", "readability", "raz-features-old.csv"))
    books_raz = pd.read_csv(books_raz_path)
    # Pulled from http://www.corestandards.org/assets/Appendix_B.pdf
    books_ccss_path = str(Path(__file__).resolve().parent.parent.parent.joinpath(
        "datasets", "readability", "ccss_text_exemplars.csv"))
    books_ccss = pd.read_csv(books_ccss_path)
    books_fcs_path = Path(__file__).resolve().parent.parent.parent.joinpath(
        "datasets", "readability", "FreeChildrenStories.csv")
    books_fcs = pd.read_csv(books_fcs_path)
    books_raz_no_na = books_raz[books_raz["Grade"].notnull()]
    books_raz_min = books_raz_no_na[["title", "Grade", "Text"]].copy()
    books_raz_min.rename(columns={"Grade": "grade", "Text": "text"}, inplace=True)
    books_ccss_min = books_ccss[["title", "text", "grade_min"]].copy()
    books_ccss_min.rename(columns={"grade_min": "grade"}, inplace=True)
    books_fcs_min = books_fcs[["title", "grade", "text"]].copy()
    books = pd.concat([books_ccss_min, books_raz_min, books_fcs_min], ignore_index=True)
    tqdm.pandas(desc="Converting grade labels to numeric")
    books["grade_numeric"] = books.progress_apply(numbify_grades, axis=1)

    return books


def load_web():
    """
    Function to load the datasets of web resources to be analyzed for readability levels.
    """
    # TODO Add the WeeBit dataset to this
    # weebit_data = None
    # failed_files = []
    # for level in ["WRLevel2", "WRLevel3", "WRLevel4"]:
    #     level_dir = Path(__file__).resolve().parent.parent.parent.joinpath("datasets", "readability", "WeeBitCorpus",
    #                                                                         "WeeBit-TextOnly", level)
    #     for file_item in tqdm(level_dir.iterdir(), desc=f"Loading files from {level_dir}"):
    #         try:
    #             try:
    #                 file_frame = pd.read_csv(file_item, names=["text"], encoding="us-ascii")
    #             except UnicodeDecodeError:
    #                 try:
    #                     file_frame = pd.read_csv(file_item, names=["text"], encoding="iso-8859-1")
    #                 except UnicodeDecodeError:
    #                     failed_files.append(file_item)
    #                     continue
    #         except pd.errors.ParserError:
    #             failed_files.append(file_item)
    #             continue
    #         file_frame["grade"] = int(level.replace("WRLevel", ""))
    #         file_frame["title"] = ""
    #         weebit_data = pd.concat([weebit_data, file_frame])
    # print(f"Failed to read {len(failed_files)} WeeBit files. :(")
    # # clean up null text fields
    # weebit_data.dropna(inplace=True)

    # web_path = Path(__file__).resolve().parent.parent.parent.joinpath(
    #     "datasets", "readability", "newsela_english.csv")
    # web = pd.read_csv(web_path)
    # web.rename(columns={"grade_level": "grade"}, inplace=True)

    web_path = Path(__file__).resolve().parent.parent.parent.joinpath(
        "datasets", "readability", "web_readability_content.csv")
    web = pd.read_csv(web_path)
    web.rename(columns={"content_new": "text"}, inplace=True)
    web_pt2_path = Path(__file__).resolve().parent.parent.parent.joinpath(
        "datasets", "readability", "TimesForKids.csv")
    web_pt2 = pd.read_csv(web_pt2_path)
    web_pt1 = web[["text", "grade", "title"]].copy()
    web_pt2 = web_pt2[["text", "grade", "title"]].copy()
    # web = pd.concat([web_pt1, web_pt2, weebit_data], ignore_index=True)
    web = pd.concat([web_pt1, web_pt2], ignore_index=True)
    tqdm.pandas(desc="Converting grade labels to numeric")
    web["grade_numeric"] = web.progress_apply(numbify_grades, axis=1)
    # web = web[web.text.str.len() > 75]

    # print("WeeBit:\t", weebit_data.shape)
    print("NewsELA:\t", web_pt1.shape)
    print("Times For Kids:\t", web_pt2.shape)

    return web


def numbify_grades(row):
    """
    Function to convert categorical reading level labels to numerical values for use in metric calculations
    """
    if row["grade"] == "K" or row["grade"] == "3 to 5" or row["grade"] == "k-1" or row["grade"] == "K-1":
        return 0.0
    elif row["grade"] == "5 to 8":
        return 1.0
    elif row["grade"] == "5+":
        return 5.0
    elif row["grade"] == "CCR":
        return 13.0
    else:
        return float(row["grade"])
    
    
def plot_error_rate(frame, spache_output, trad_output, purpose="thesis"):
    """
    Function that utilizes a dataframe of error values to generate a box plot for visual analysis of formula
    performances
    """
    spache_original = frame[["grade_numeric", "spache_original_er"]]
    spache_original = spache_original.assign(Formula="Spache")
    spache_original.rename(columns={"spache_original_er": "error_rate"}, inplace=True)

    spache_allen = frame[["grade_numeric", "spache_allen_er"]]
    spache_allen = spache_allen.assign(Formula="Spache-Allen")
    spache_allen.rename(columns={"spache_allen_er": "error_rate"}, inplace=True)

    spache_sven = frame[["grade_numeric", "spache_sven_er"]]
    spache_sven = spache_sven.assign(Formula="Spache (Sven)")
    spache_sven.rename(columns={"spache_sven_er": "error_rate"}, inplace=True)

    if purpose == "thesis":
        spache_aoa_11 = frame[["grade_numeric", "spache_aoa_11_er"]]
        spache_aoa_11 = spache_aoa_11.assign(Formula="Spache (AoA < 11)")
        spache_aoa_11.rename(columns={"spache_aoa_11_er": "error_rate"}, inplace=True)

        spache_aoa_full = frame[["grade_numeric", "spache_aoa_full_er"]]
        spache_aoa_full = spache_aoa_full.assign(Formula="Spache (AoA Full)")
        spache_aoa_full.rename(columns={"spache_aoa_full_er": "error_rate"}, inplace=True)

        spache_sven_aoa_11 = frame[["grade_numeric", "spache_sven_aoa_11_er"]]
        spache_sven_aoa_11 = spache_sven_aoa_11.assign(Formula="Spache (Sven + AoA < 11)")
        spache_sven_aoa_11.rename(columns={"spache_sven_aoa_11_er": "error_rate"}, inplace=True)
        # generate and save image
        er_frame = spache_original.append(
            [spache_aoa_11, spache_sven, spache_aoa_full, spache_sven_aoa_11, spache_allen])

        # size = 22
        # plt_params = {
        #     'legend.title_fontsize': 14,
        #     'legend.fontsize': size,
        #     'axes.labelsize': size,
        #     'xtick.labelsize': 18,
        #     'ytick.labelsize': 18
        # }
        # matplotlib.rcParams.update(plt_params)

        plt.figure(figsize=FIGURE_SIZE)
        img = sns.boxplot(y='error_rate', x='grade_numeric', data=er_frame, palette="colorblind", hue='Formula')
        img.set(xlabel="Grade Level", ylabel="Error Rate")
        fig = img.get_figure()
        fig.savefig(spache_output)
        print(f"Results for Spache extensions saved to {spache_output}")

    # traditional formulas
    fkg = frame[["grade_numeric", "fkg_er"]]
    fkg = fkg.assign(Formula="Flesch-Kincaid")
    fkg.rename(columns={"fkg_er": "error_rate"}, inplace=True)

    cli = frame[["grade_numeric", "cli_er"]]
    cli = cli.assign(Formula="Coleman-Liau Index")
    cli.rename(columns={"cli_er": "error_rate"}, inplace=True)

    lix = frame[["grade_numeric", "lix_er"]]
    lix = lix.assign(Formula="LIX Readability Formula")
    lix.rename(columns={"lix_er": "error_rate"}, inplace=True)

    rix = frame[["grade_numeric", "rix_er"]]
    rix = rix.assign(Formula="RIX Readability Formula")
    rix.rename(columns={"rix_er": "error_rate"}, inplace=True)

    smog = frame[["grade_numeric", "smog_er"]]
    smog = smog.assign(Formula="SMOG")
    smog.rename(columns={"smog_er": "error_rate"}, inplace=True)

    gfog = frame[["grade_numeric", "gfog_er"]]
    gfog = gfog.assign(Formula="Gunning-Fog Index")
    gfog.rename(columns={"gfog_er": "error_rate"}, inplace=True)

    dale_chall = frame[["grade_numeric", "dale_chall_er"]]
    dale_chall = dale_chall.assign(Formula="New Dale-Chall")
    dale_chall.rename(columns={"dale_chall_er": "error_rate"}, inplace=True)

    if purpose == "thesis":
        er_frame = fkg.append([spache_original, cli, lix, rix, smog, gfog, dale_chall])
    else:
        er_frame = fkg.append([spache_original, spache_sven, spache_allen, cli, lix, rix, smog, gfog, dale_chall])

    # Save image
    plt.figure(figsize=FIGURE_SIZE)
    img = sns.boxplot(y='grade_numeric', x='error_rate', data=er_frame, palette="colorblind", hue='Formula', orient='h')
    img.set_ylabel("Grade Level", fontsize=48)
    img.set_xlabel("Error Rate", fontsize=48)
    img.tick_params(labelsize=36)
    # img.set(xlabel="Grade Level", ylabel="Error Rate")
    fig = img.get_figure()
    fig.savefig(trad_output)
    print(f"Results for traditional formulas saved to {trad_output}")
    

def run_books(purpose="thesis"):
    """
    Driver function for executing the readability experiment on the books dataset(s)
    """
    print("Running data for BOOKS")
    print("=" * 75)
    # BOOKS
    books = load_books()

    # readability levels
    books = assess_readability(books, purpose)

    # error rate
    books = calculate_error_rate(books, "books", purpose)

    # boxplot
    if purpose == "thesis":
        spache_path = Path(__file__).resolve().parent.parent.parent.joinpath(
            "datasets", "readability", "images", f"{FIGURE_SIZE[0]}x{FIGURE_SIZE[1]}",
            f"spache-er-comparison.png")
        trad_path = Path(__file__).resolve().parent.parent.parent.joinpath(
            "datasets", "readability", "images", f"{FIGURE_SIZE[0]}x{FIGURE_SIZE[1]}",
            f"formulas-er-comparison-spache.png")
        plot_error_rate(books, spache_path, trad_path, purpose)
    else:
        trad_path = Path(__file__).resolve().parent.parent.parent.joinpath(
            "datasets", "readability", "images", f"{FIGURE_SIZE[0]}x{FIGURE_SIZE[1]}",
            f"thesis-formulas-comparison-books.png")
        plot_error_rate(books, None, trad_path, purpose)


def run_experiment(purpose="thesis"):
    """
    Driver function for the entirety of the readabilty experiment.
    """
    Path(str(Path(__file__).resolve().parent.parent.parent.joinpath(
        "datasets", "readability", "images", f"{FIGURE_SIZE[0]}x{FIGURE_SIZE[1]}"))).mkdir(parents=True, exist_ok=True)
    run_books(purpose)
    run_web(purpose)


def run_web(purpose="thesis"):
    """
    Driver function to execute the web resources portion of the readability experiment.
    """
    print("Running data for WEB")
    print("=" * 75)
    # WEB
    web = load_web()

    # readability levels
    web = assess_readability(web, purpose)

    # error rate analysis
    web = calculate_error_rate(web, "web", purpose)

    # boxplot
    if purpose == "thesis":
        spache_path = Path(__file__).resolve().parent.parent.parent.joinpath(
            "datasets", "readability", "images", f"{FIGURE_SIZE[0]}x{FIGURE_SIZE[1]}",
            f"spache-web-er-comparison.png")
        trad_path = Path(__file__).resolve().parent.parent.parent.joinpath(
            "datasets", "readability", "images", f"{FIGURE_SIZE[0]}x{FIGURE_SIZE[1]}",
            f"formulas-web-er-comparison-spache.png")
        plot_error_rate(web, spache_path, trad_path, purpose)
    else:
        trad_path = Path(__file__).resolve().parent.parent.parent.joinpath(
            "datasets", "readability", "images", f"{FIGURE_SIZE[0]}x{FIGURE_SIZE[1]}",
            f"thesis-formulas-comparison-web.png")
        plot_error_rate(web, None, trad_path, purpose)


if __name__ == "__main__":
    size = 22
    plt_params = {
    #     'legend.title_fontsize': 14,
        'legend.fontsize': size,
    #     'axes.labelsize': size,
    #     'xtick.labelsize': 18,
    #     'ytick.labelsize': 18
    }
    matplotlib.rcParams.update(plt_params)
    tqdm.pandas()
    purpose = "thesis"
    # purpose = "ecir"
    run_experiment(purpose)
