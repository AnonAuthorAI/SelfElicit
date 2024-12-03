import re
import pandas as pd
from qa_metrics.em import em_match
from qa_metrics.f1 import f1_match, f1_score_with_precision_recall


def evaluate(true_ans_list, model_ans, sel_metric="f1"):
    best_score = -1
    for true_ans in true_ans_list:
        eval_res = evaluate_single_ans(true_ans, model_ans)
        if eval_res[sel_metric] > best_score:
            best_score = eval_res[sel_metric]
            best_ans = true_ans
            best_eval_res = eval_res

    best_eval_res["em"] = em_match(true_ans_list, model_ans)
    return best_ans, best_eval_res


def evaluate_single_ans(true_ans, model_ans):
    f1_pr = f1_score_with_precision_recall(true_ans, model_ans)
    f1, pr, re = f1_pr["f1"], f1_pr["precision"], f1_pr["recall"]
    return {
        "em": em_match(true_ans, model_ans),
        "f1m": f1_match(true_ans, model_ans),
        "f1": f1,
        "pr": pr,
        "re": re,
    }


def add_new_eval_metrics(df_res):
    df_new_eval = []

    for idx in range(len(df_res)):
        row = df_res.iloc[idx]
        em = em_match(row["true_ans"], row["model_ans"])
        f1m = f1_match(row["true_ans"], row["model_ans"])
        f1_pr = f1_score_with_precision_recall(row["true_ans"], row["model_ans"])
        f1, pr, re = f1_pr["f1"], f1_pr["precision"], f1_pr["recall"]
        df_new_eval.append(
            [row["model_ans"], row["true_ans"], row["is_correct"], em, f1m, f1, pr, re]
        )

    df_new_eval = pd.DataFrame(
        df_new_eval,
        columns=["model_ans", "true_ans", "is_correct", "em", "f1m", "f1", "pr", "re"],
    )
    for metric in ["em", "f1m", "f1", "pr", "re"]:
        df_res[metric] = df_new_eval[metric]

    return df_res


def remove_helpers_and_symbols(text):
    helpers_pattern_start = r"^(a|an|the)\s+"
    helpers_pattern_end = r"\s+(a|an|the)$"
    symbols_pattern_start = r"^[\ ,\.\:\"\']+"
    symbols_pattern_end = r"[\ ,\.\:\"\']+$"

    while True:
        new_text = re.sub(symbols_pattern_start, "", text)
        new_text = re.sub(helpers_pattern_start, "", new_text, flags=re.IGNORECASE)

        if new_text == text:
            break

        text = new_text

    while True:
        new_text = re.sub(symbols_pattern_end, "", text)
        new_text = re.sub(helpers_pattern_end, "", new_text, flags=re.IGNORECASE)

        if new_text == text:
            break

        text = new_text

    return text.strip()
