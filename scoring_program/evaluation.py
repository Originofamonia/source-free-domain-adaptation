import difflib
import glob
import sys
import os
import fnmatch
import collections
from sklearn.metrics import f1_score,precision_score,recall_score
from anafora import evaluate, timeml


def path_lines(root, subdir, pattern, replace=None):
    result = []
    prefix = os.path.join(root, subdir)
    for name in glob.glob(os.path.join(prefix, pattern), recursive=True):
        if os.path.isfile(name):
            name = name[len(prefix) + 1:]
            if replace is not None:
                for old, new in replace.items():
                    name = name.replace(old, new)
            result.append(name + '\n')
    return result


def score_time(ref_domain, res_domain, results):
    scores_type=evaluate.Scores
    exclude=("Event")
    file_named_scores = evaluate.score_dirs(
        reference_dir=ref_domain,
        predicted_dir=res_domain,
        exclude=exclude) # pairwise=True

    all_named_scores = collections.defaultdict(lambda: scores_type())
    for _, named_scores in file_named_scores:
        for name, scores in named_scores.items():
            all_named_scores[name].update(scores)

    results['time_f1']= all_named_scores["*"].f1()
    results['time_precision'] = all_named_scores["*"].precision()
    results['time_recall'] = all_named_scores["*"].recall()

def score_negation(ref_domain,res_domain,results):
    ref, res = read_tsvs(ref_domain, res_domain)
    assert len(ref) == len(res)
    results['negation_f1']=  f1_score(ref,res)
    results['negation_precision'] = precision_score(ref,res)
    results['negation_recall'] = recall_score(ref,res)


def read_tsvs(ref_file, res_file):
    ref_output = []
    res_output = []
    with open(ref_file, 'r') as f_ref_output, open(res_file, 'r') as f_res_output:
        for ref_record, res_record in zip(f_ref_output, f_res_output):
            ref_record = int(ref_record)
            if ref_record != 0:
                ref_output.append(ref_record)
                res_output.append(int(res_record))
    return ref_output, res_output


def main():
    print("SemEval 2021 Task 10 scorer v0.2")
    # _, input_dir, output_dir = sys.argv
    # input_dir = os.path.join(os.getcwd(), f"outputs")
    input_dir = os.path.join(os.getcwd(), f'practice_data')
    output_dir = os.path.join(os.getcwd(), f"outputs")

    # check which tasks have been submitted
    has_time = os.path.exists(os.path.join(input_dir, 'res', 'time'))
    has_negation = os.path.exists(os.path.join(input_dir, 'res', 'negation'))
    replaces = {'gold': 'system', 'fake': 'TimeNorm.system.completed.xml'}

    # exit with an error if any of the expected files were not submitted
    if has_time == has_negation:  # has both or has neither
        expected = path_lines(input_dir, 'ref', "**", replaces)
    elif has_time:
        expected = path_lines(input_dir, 'ref', "time/**", replaces)
    else:  # has_negation
        expected = path_lines(input_dir, 'ref', "negation/**", replaces)
    uploaded = path_lines(input_dir, 'res', "**")
    diff = list(difflib.unified_diff(a=expected, b=uploaded, n=0,
                                     fromfile="expected", tofile="uploaded"))
    if diff:
        sys.stderr.write("Incorrect files:\n")
        sys.stderr.writelines(diff)
        sys.exit(1)

    # scoring
    metrics = {}
    if has_time:
        ref_domain = os.path.join(input_dir, 'ref', 'time')
        res_domain = os.path.join(input_dir, 'res', 'time')
        score_time(ref_domain, res_domain, metrics)
    if has_negation:
        ref_domain = os.path.join(input_dir, 'ref', 'negation/gold.tsv')
        res_domain = os.path.join(input_dir, 'res', 'negation/system.tsv')
        score_negation(ref_domain, res_domain, metrics)

    # write scores file
    with open(os.path.join(output_dir, "scores.txt"), "w") as output_file:
        for key, value in metrics.items():
            output_file.write(f"{key}:{value}\n")

    print("Success!")


if __name__ == "__main__":
    main()
