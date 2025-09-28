
import csv
from collections import Counter

ALLOWED_LABELS = {"arthm", "dos", "le", "rent", "tm", "too", "ue"}

def parse_labels(s: str):
    if not s:
        return set()
    return {x.strip().lower() for x in s.split(",") if x.strip()}

def main():
    INPUT_PATH = "./results/ESBMC_7.9_RESULTS.csv"

    tp, fp, fn = Counter(), Counter(), Counter()
    num_exact = num_hit_nonempty = total = 0

    with open(INPUT_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            pred_set = parse_labels(row["pred_labels"])
            gt_set   = parse_labels(row["gt_labels"])
            total += 1

            if pred_set == gt_set:
                num_exact += 1
            if pred_set & gt_set:
                num_hit_nonempty += 1

            for l in gt_set:
                if l not in pred_set:
                    fn[l] += 1
            for l in pred_set:
                if l not in gt_set:
                    fp[l] += 1
            for l in (pred_set & gt_set):
                tp[l] += 1

    # ====== Per-label ======
    print("Per-label metrics:")
    all_labels = sorted(ALLOWED_LABELS & (set(tp) | set(fp) | set(fn)))
    precisions, recalls, f1s = [], [], []

    for l in all_labels:
        TP, FP, FN = tp[l], fp[l], fn[l]
        P = TP / (TP + FP) if (TP + FP) else 0
        R = TP / (TP + FN) if (TP + FN) else 0
        F1 = 2*P*R/(P+R) if (P+R) else 0
        precisions.append(P); recalls.append(R); f1s.append(F1)
        print(f"{l.upper():5s} | P={P:.2f} R={R:.2f} F1={F1:.2f} | TP={TP} FP={FP} FN={FN}")

    # ====== Global ======
    macro_P = sum(precisions)/len(precisions) if precisions else 0
    macro_R = sum(recalls)/len(recalls) if recalls else 0
    macro_F1 = sum(f1s)/len(f1s) if f1s else 0

    total_TP, total_FP, total_FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    micro_P = total_TP / (total_TP + total_FP) if (total_TP+total_FP) else 0
    micro_R = total_TP / (total_TP + total_FN) if (total_TP+total_FN) else 0
    micro_F1 = 2*micro_P*micro_R/(micro_P+micro_R) if (micro_P+micro_R) else 0

    print("\nGlobal metrics:")
    print(f" - Exact Match Acc (EMA): {num_exact/total:.3f}")
    print(f" - At Least One Correct (ONE): {num_hit_nonempty/total:.3f}")
    print(f" - Macro Avg: P={macro_P:.2f}, R={macro_R:.2f}, F1={macro_F1:.2f}")
    print(f" - Micro Avg: P={micro_P:.2f}, R={micro_R:.2f}, F1={micro_F1:.2f}")

if __name__ == "__main__":
    main()

