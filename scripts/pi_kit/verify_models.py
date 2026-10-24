"""Check that the trained models on this device reproduce the predictions the project evaluated.

Runs the 256 real test windows in check/ through the three TFLite branches, fuses them with the
rule in models/manifest.json, and compares with check/expected.npy (labels) and
check/expected_probabilities.npy. Writes verify_pi.json. Anything below 100% agreement means the
device is not running the evaluated model, and no timing from it should be reported.
"""
import json, os, platform
import numpy as np

def interp():
    try:
        from ai_edge_litert.interpreter import Interpreter; return Interpreter, "ai_edge_litert"
    except ImportError:
        from tflite_runtime.interpreter import Interpreter; return Interpreter, "tflite_runtime"

here = os.path.dirname(os.path.abspath(__file__))
man = json.load(open(os.path.join(here, "models", "manifest.json")))
X = np.load(os.path.join(here, "check", "sample_windows.npy")).astype(np.float32)
lab = np.load(os.path.join(here, "check", "expected.npy"))
prob = np.load(os.path.join(here, "check", "expected_probabilities.npy"))
Interp, backend = interp()
ps = []
for b in man["branches"]:
    it = Interp(model_path=os.path.join(here, "models", b["file"])); it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    out = []
    for x in X:
        it.set_tensor(i["index"], x[None]); it.invoke(); out.append(it.get_tensor(o["index"])[0].copy())
    ps.append(np.array(out))
ps = np.stack(ps)
g = man["fusion"]["gamma"]; eps = man["fusion"]["epsilon"]
pri = np.array([man["fusion"]["branch_priors"][b["name"]] for b in man["branches"]])[:, None]
c = pri * ps.max(axis=2) ** g
w = c / (c.sum(axis=0) + eps)
fused = (w[:, :, None] * ps).sum(axis=0)
agree = float((fused.argmax(1) == lab).mean())
diff = float(np.abs(fused - prob).max())
board = ""
for p in ("/proc/device-tree/model",):
    if os.path.exists(p):
        board = open(p, errors="ignore").read().strip("\x00").strip()
res = {"board": board or platform.machine(), "backend": backend, "windows": int(len(X)),
       "argmax_agreement": agree, "max_probability_difference": diff,
       "pass": agree == 1.0 and diff < 1e-4}
json.dump(res, open(os.path.join(here, "verify_pi.json"), "w"), indent=1)
print(f"Model check on {res['board']}: agreement {agree*100:.2f}% over {len(X)} real test windows, "
      f"max probability difference {diff:.2e}  ->  {'PASS' if res['pass'] else 'FAIL'}")
