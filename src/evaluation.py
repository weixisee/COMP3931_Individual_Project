import numpy as np
from sklearn.metrics import roc_curve, recall_score, roc_auc_score

# find the threshold that have the most balanced fpr and tpr
def youden_threshold(y_true, y_prob):
    fpr_value, tpr_value, thresholds = roc_curve(y_true, y_prob)

    j_scores = tpr_value - fpr_value
    youden_index = np.argmax(j_scores)
    optimal_threshold = thresholds[youden_index]

    return optimal_threshold

# find the tpr disparity between male and female
def gender_disparity(y_prob, y_true, df, threshold):

    # initialise a dictionary to store the result 
    disparities = {}

    # turn the probability into binary classification (0/1)
    y_pred = (y_prob >= threshold).astype(int)

    male_group = (df['tasks/patient sex'] == 1).to_numpy()
    female_group = (df['tasks/patient sex'] == 0).to_numpy()

    # group them based on male or female
    male_preds = y_pred[male_group]
    male_true = y_true[male_group]

    female_preds = y_pred[female_group]
    female_true = y_true[female_group]

    male_tpr = recall_score(male_true, male_preds)
    female_tpr = recall_score(female_true, female_preds)

    tpr_disparity = male_tpr - female_tpr

    disparities['atelectasis'] = {
        'tpr_male': round(float(male_tpr), 4),
        'tpr_female': round(float(female_tpr), 4),
        'tpr_disparity': round(float(tpr_disparity), 4),
        "male_samples": int(male_group.sum()),
        "female_samples": int(female_group.sum()),
        "total_samples": int(len(y_pred))
    }

    return disparities

# find the overall auroc between male and female
def auc_by_sex(y_prob, y_true, df):

    results = {}

    male_group = (df['tasks/patient sex'] == 1).to_numpy()
    female_group = (df['tasks/patient sex'] == 0).to_numpy()

    male_true = y_true[male_group]
    male_prob = y_prob[male_group]

    female_true = y_true[female_group]
    female_prob = y_prob[female_group]

    auc_male = roc_auc_score(male_true, male_prob)
    auc_female = roc_auc_score(female_true, female_prob)

    auc_gap = auc_male - auc_female

    results["atelectasis"] = {
    "auc_male": round(float(auc_male), 4),
    "auc_female": round(float(auc_female), 4),
    "auc_gap": round(float(auc_gap), 4),
    "male_samples": int(male_group.sum()),
    "female_samples": int(female_group.sum()),
    "total_samples": int(len(y_true))
  }

    return results

# find the age disparity bewteen the age splits chosen (50 years old)
def age_disparity(y_prob, y_true, df, threshold):

  disparities = {}

  y_pred = (y_prob >= threshold).astype(int)

  young = (df['patient_age'] < 50).to_numpy()
  old = (df['patient_age'] >= 50).to_numpy()

  young_true = y_true[young]
  young_pred = y_pred[young]

  old_true = y_true[old]
  old_pred = y_pred[old]

  young_tpr = recall_score(young_true, young_pred)
  old_tpr = recall_score(old_true, old_pred)

  tpr_disparity = old_tpr - young_tpr

  disparities["atelectasis"] = {
    'tpr_young': round(float(young_tpr), 4),
    'tpr_old': round(float(old_tpr), 4),
    'tpr_disparity': round(float(tpr_disparity), 4),
    'young_samples': int(young.sum()),
    'old_samples': int(old.sum()),
    'total_samples': int(len(y_pred))
  }

  return disparities

def auc_by_age(y_prob, y_true, df):

    results = {}

    young = (df['patient_age'] < 50).to_numpy()
    old = (df['patient_age'] >= 50).to_numpy()

    young_true = y_true[young]
    young_prob = y_prob[young]

    old_true = y_true[old]
    old_prob = y_prob[old]

    auc_young = roc_auc_score(young_true, young_prob)
    auc_old = roc_auc_score(old_true, old_prob)

    auc_gap = auc_old - auc_young

    results["atelectasis"] = {
    "auc_young": round(float(auc_young), 4),
    "auc_old": round(float(auc_old), 4),
    "auc_gap": round(float(auc_gap), 4),
    'young_samples': int(young.sum()),
    'old_samples': int(old.sum()),
    'total_samples': int(len(y_true))
  }

    return results

# tpr for intersectional group
def intersectional_disparity(y_prob, y_true, df, threshold):

    disparities = {}

    y_pred = (y_prob >= threshold).astype(int)

    # define groups
    male = (df['tasks/patient sex'] == 1).to_numpy()
    female = (df['tasks/patient sex'] == 0).to_numpy()

    young = (df['patient_age'] < 50).to_numpy()
    old = (df['patient_age'] >= 50).to_numpy()


    groups = {
        "male_<50": male & young,
        "male_>=50": male & old, 
        "female_<50": female & young, 
        "female_>=50": female & old
    }

    tprs = {}

    # compute TPR per group
    for name, mask in groups.items():
        group_true = y_true[mask]
        group_pred = y_pred[mask]

        tpr = recall_score(group_true, group_pred)

        tprs[name] = tpr

    # compute median TPR
    median_tpr = np.median(list(tprs.values()))

    # compute disparities by find the difference of each group with the median tpr
    gaps = {}
    for k, v in tprs.items():
        gaps[k] = v - median_tpr

    # overall intersectional gap (max - min)
    max_min_gap = max(tprs.values()) - min(tprs.values())
   
    disparities["atelectasis"] = {
        "tpr_per_group": {k: round(float(v), 4) for k, v in tprs.items()},
        "median_tpr": round(float(median_tpr), 4),
        "gap_per_group": {k: round(float(v), 4) for k, v in gaps.items()},
        "max_min_tpr_gap": round(float(max_min_gap), 4),
        "group_sizes": {k: int(mask.sum()) for k, mask in groups.items()}
    }

    return disparities

def auc_by_intersection(y_prob, y_true, df):

    results = {}

    male = (df['tasks/patient sex'] == 1).to_numpy()
    female = (df['tasks/patient sex'] == 0).to_numpy()

    young = (df['patient_age'] < 50).to_numpy()
    old = (df['patient_age'] >= 50).to_numpy()

    groups = {
        "male_<50": male & young,
        "male_>=50": male & old,
        "female_<50": female & young,
        "female_>=50": female & old
    }

    aucs = {}

    for name, mask in groups.items():
        group_true = y_true[mask]
        group_prob = y_prob[mask]

        auc = roc_auc_score(group_true, group_prob)

        aucs[name] = auc

    median_auc = np.median(list(aucs.values()))

    # disparity from median
    auc_gaps = {}
    for k, v in aucs.items():
        auc_gaps[k] = v - median_auc
    
    max_min_gap = max(aucs.values()) - min(aucs.values())

    results["atelectasis"] = {
        "auc_per_group": {k: round(float(v), 4) for k, v in aucs.items()},
        "median_auc": round(float(median_auc), 4),
        "auc_gap_per_group": { k: round(float(v), 4) for k, v in auc_gaps.items()},
        "max_min_auc_gap": round(float(max_min_gap), 4),
        "group_sizes": {k: int(mask.sum()) for k, mask in groups.items()}
    }

    return results
