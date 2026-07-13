import json
import sys

def flatten_metrics(rep):
    m = {}
    for side, metrics in rep.get("metrics_by_side", {}).items():
        for metric in metrics:
            m[f"{side}_{metric['name']}"] = metric['value']
    return m

def parse_val(val_str):
    parts = val_str.split("/")
    return [float("".join(c for c in p if c.isdigit() or c in ".-")) for p in parts if any(c.isdigit() for c in p)]

def compare_reports(old_path, new_path):
    with open(old_path, "r") as f:
        old_rep = json.load(f)
    with open(new_path, "r") as f:
        new_rep = json.load(f)
        
    old_m = flatten_metrics(old_rep)
    new_m = flatten_metrics(new_rep)
    
    for key in sorted(set(old_m.keys()) | set(new_m.keys())):
        old_val = str(old_m.get(key, "N/A"))
        new_val = str(new_m.get(key, "N/A"))
        
        diff_str = ""
        try:
            o_f = parse_val(old_val)
            n_f = parse_val(new_val)
            
            if len(o_f) == len(n_f) and len(o_f) > 0:
                diffs = [n - o for n, o in zip(n_f, o_f)]
                diff_str = " | Diff: " + " / ".join([f"{d:+.2f}" for d in diffs])
                if any(abs(d) > 2.0 for d in diffs):
                    diff_str += " (WARNING: >2.0mm/cm3 drift!)"
        except Exception:
            pass
            
        print(f"{key:35s}: {old_val:>10s} -> {new_val:>10s}{diff_str}")

if __name__ == "__main__":
    compare_reports(sys.argv[1], sys.argv[2])
