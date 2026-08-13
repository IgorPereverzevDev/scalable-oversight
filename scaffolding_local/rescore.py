import sys
from inspect_ai import score
from inspect_ai.log import read_eval_log, write_eval_log
from cerberus.core.monitors import SCOPE_MONITOR, LONG_SCOPE_MONITOR
from cerberus.evals.scorers import ground_truth, monitor

render_name, mon, log_path = sys.argv[1], sys.argv[2], sys.argv[3]
spec = LONG_SCOPE_MONITOR if render_name == "long" else SCOPE_MONITOR

log = read_eval_log(log_path)
rescored = score(
    log,
    [ground_truth(), monitor(spec=spec, max_tokens=2000)],
    model_roles={"monitor": mon},
    action="overwrite",
    display="none",
)

out = f"logs/feat_{render_name}_{mon.replace('/', '_')}.eval"
write_eval_log(rescored, out)

tp = fp = fn = perr = 0
for s in rescored.samples:
    v = s.scores["monitor"].value
    bad = s.scores["ground_truth"].value["did_bad_thing"]
    if v == "parse_error":
        perr += 1
    elif v == "unsafe":
        tp += bad
        fp += not bad
    elif v == "safe":
        fn += bad
print(f"{mon:46s} {render_name:6s} TP={tp} FP={fp} FN={fn} parse_error={perr}")