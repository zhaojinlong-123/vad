import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

if __name__ == "__main__":
    # Ground truth speech label
    gt = pd.read_csv("competition_dataset/test_labels.csv").sort_values(["id", "s"])
    uids = list(set(gt["id"]))
    # Remove the No-Speech labels, and only keep the Speech labels
    gt = gt[gt["label_index"] != 3]
    # Prediction
    pre = pd.read_csv("test_pre_7.csv").sort_values(["id", "s"])

    # For different shifting window, we get different frames in each second.
    # We will use the max F1 for the final result
    frames = [100, 50, 40, 20, 10]  # 10ms, 20ms, 25ms, 50ms, 100ms
    f1_scores = []
    for f in frames:
        gt_labels = np.zeros((900*len(uids)*f,))
        pre_labels = np.zeros((900*len(uids)*f,))
        # Fill the predicted speech label with 1
        for u in uids:
            utt_start_frames = 900*uids.index(u)*f

            gt_utt = gt[gt["id"] == u]
            for i in range(len(gt_utt)):
                df = gt_utt.iloc[i]
                start_frame = int(df["s"] * f)
                end_frame = int(df["e"] * f)
                assert start_frame <= end_frame
                gt_labels[utt_start_frames + start_frame: utt_start_frames + end_frame] = 1

            pre_utt = pre[pre["id"] == u]
            for i in range(len(pre_utt)):
                df = pre_utt.iloc[i]
                start_frame = int(df["s"] * f)
                end_frame = int(df["e"] * f)
                assert start_frame <= end_frame
                pre_labels[utt_start_frames + start_frame: utt_start_frames + end_frame] = 1

        # tp = ((gt_labels == pre_labels) & (pre_labels == 1)).sum()
        # fp = ((gt_labels != pre_labels) & (pre_labels == 1)).sum()
        # fn = ((gt_labels != pre_labels) & (pre_labels == 0)).sum()
        # precision = tp / (tp + fp)
        # recall = tp / (tp + fn)
        # f1 = 2 * precision * recall / (precision + recall)

        f1_scores.append(f1_score(gt_labels, pre_labels))
    print(np.asarray(f1_scores).max())

