"""Phase 5.1: measure per-window inference cost of the frozen ensemble architecture.

Runs on ANY machine (this x86 VM, the faculty's Raspberry Pi, a Jetson). It records the host it ran
on from the OS, never from a flag, so a number can never travel without its hardware.

What it measures, per model variant (cnn, bilstm, transformer, full ensemble = 3 branches + fusion):
  - latency per window at batch 1: p50, p90, p95, p99 (ms), after warm-up
  - latency under load: the same, while k background CPU-burner processes run (k = 0..cores)
  - sustained throughput (windows/s) and CPU utilisation of the inference process during it
  - memory: resident set size after loading each model, and the .tflite file size
  - energy: read from RAPL (/sys/class/powercap) if present; otherwise reported as NOT MEASURED.
    On a Pi, energy needs an external USB power meter; enter it with --meter-watts.

Inference cost does not depend on weight values, so the branches are built from the frozen
architecture (docs/architecture_decision.md) with seeded random weights when --model-dir is not
given. On the Pi, pass --model-dir with the trained export (artifacts/deployment/float32) instead.

Usage:
  python3 measure_latency.py --out results/latency_x86.json
  python3 measure_latency.py --model-dir deployment/float32 --out latency_pi.json   # on the Pi
"""
import argparse
import json
import multiprocessing as mp
import os
import platform
import time

import numpy as np

SEQ, NFEAT = 10, 62


def host_info():
    model = None
    for p in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        if os.path.exists(p):
            model = open(p, errors="ignore").read().strip("\x00").strip()
    cpu = ""
    if os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith(("model name", "hardware")):
                cpu = line.split(":", 1)[1].strip()
                break
    mem = 0.0
    if os.path.exists("/proc/meminfo"):
        mem = int(open("/proc/meminfo").read().split("MemTotal:")[1].split()[0]) / 1024 / 1024
    return {"board": model or f"{platform.system()} {platform.machine()} (not a Raspberry Pi)",
            "cpu": cpu, "arch": platform.machine(), "cores": os.cpu_count(), "ram_gb": round(mem, 2),
            "os": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
            "is_raspberry_pi": bool(model and "raspberry pi" in model.lower())}


def build_branches():
    """Frozen architecture, copied from cnn_branch.py, bilstm_branch.py, transformer_branch.py."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    seed = 42
    init = keras.initializers.GlorotUniform(seed=seed)

    def cnn():
        i = keras.Input((SEQ, NFEAT))
        x = i
        for k, f in enumerate((64, 128, 64), 1):
            x = layers.Conv1D(f, 3, padding="same", activation="relu", kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.GlobalMaxPooling1D()(x)
        return keras.Model(i, layers.Dense(2, activation="softmax", kernel_initializer=init)(x), name="cnn")

    def bilstm():
        i = keras.Input((SEQ, NFEAT))
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, kernel_initializer=init, unroll=True))(i)
        x = layers.Dropout(0.3)(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, kernel_initializer=init, unroll=True))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation="relu", kernel_initializer=init)(x)
        return keras.Model(i, layers.Dense(2, activation="softmax", kernel_initializer=init)(x), name="bilstm")

    def transformer():
        pos = np.arange(SEQ, dtype=np.float32)[:, None]
        rates = 1.0 / np.power(10000.0, np.arange(0, 64, 2, dtype=np.float32) / 64)
        enc = np.zeros((SEQ, 64), np.float32)
        enc[:, 0::2] = np.sin(pos * rates)
        enc[:, 1::2] = np.cos(pos * rates)
        i = keras.Input((SEQ, NFEAT))
        x = layers.Dense(64, kernel_initializer=init)(i)
        x = x + tf.constant(enc)
        for _ in range(2):
            a = layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
            x = layers.LayerNormalization(epsilon=1e-6)(x + layers.Dropout(0.1)(a))
            f = layers.Dense(64)(layers.Dense(128, activation="relu")(x))
            x = layers.LayerNormalization(epsilon=1e-6)(x + layers.Dropout(0.1)(f))
        x = layers.GlobalAveragePooling1D()(x)
        return keras.Model(i, layers.Dense(2, activation="softmax", kernel_initializer=init)(x), name="transformer")

    return {"cnn": cnn(), "bilstm": bilstm(), "transformer": transformer()}


def export(models, outdir):
    import tensorflow as tf
    os.makedirs(outdir, exist_ok=True)
    paths, params = {}, {}
    for name, m in models.items():
        conv = tf.lite.TFLiteConverter.from_keras_model(m)
        try:
            blob = conv.convert()
        except Exception:  # noqa: BLE001
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            blob = conv.convert()
        p = os.path.join(outdir, f"{name}.tflite")
        open(p, "wb").write(blob)
        paths[name] = p
        params[name] = int(m.count_params())
    return paths, params


def interpreter_class():
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter, "ai_edge_litert"
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter, "tflite_runtime"
    except ImportError:
        import tensorflow as tf
        return tf.lite.Interpreter, "tensorflow"


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        return float("nan")


def burner(stop):
    x = 0.0
    while not stop.is_set():
        for i in range(10000):
            x += i * 1e-9


def timed(fn, n, warm):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(n):
        s = time.perf_counter_ns()
        fn()
        t.append((time.perf_counter_ns() - s) / 1e6)
    t = np.array(t)
    return {"n": n, "mean_ms": float(t.mean()), "p50_ms": float(np.percentile(t, 50)),
            "p90_ms": float(np.percentile(t, 90)), "p95_ms": float(np.percentile(t, 95)),
            "p99_ms": float(np.percentile(t, 99))}


def rapl_joules():
    base = "/sys/class/powercap"
    total, found = 0.0, False
    if os.path.isdir(base):
        for d in os.listdir(base):
            f = os.path.join(base, d, "energy_uj")
            if d.count(":") == 1 and os.path.exists(f):
                try:
                    total += int(open(f).read()) / 1e6
                    found = True
                except OSError:
                    pass
    return total if found else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None, help="directory of trained .tflite branches (cnn/bilstm/transformer)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--meter-watts", type=float, default=None, help="average watts read from an external meter")
    a = ap.parse_args()

    res = {"host": host_info(), "provenance": None}
    if a.model_dir:
        paths = {n: os.path.join(a.model_dir, f"{n}.tflite") for n in ("cnn", "bilstm", "transformer")}
        res["provenance"] = f"trained export from {a.model_dir}"
        params = {}
        meta = os.path.join(a.model_dir, "PROVENANCE.json")
        if not os.path.exists(meta):
            meta = os.path.join(a.model_dir, "FROZEN_ARCH.json")
        if os.path.exists(meta):  # pi_kit: the same frozen-architecture files measured on x86
            m = json.load(open(meta))
            res["provenance"] = m["provenance"]
            params = m.get("params", {})
    else:
        models = build_branches()
        outdir = os.path.join(os.path.dirname(os.path.abspath(a.out)), "tflite_frozen_arch")
        paths, params = export(models, outdir)
        res["provenance"] = ("frozen architecture with seeded random weights (inference cost is "
                             "independent of weight values); exported to TFLite float32")
    base_rss = rss_mb()
    Interp, backend = interpreter_class()
    res["backend"] = backend
    interps = {}
    for name, p in paths.items():
        it = Interp(model_path=p, num_threads=1)
        it.allocate_tensors()
        interps[name] = it
    res["memory"] = {"rss_before_models_mb": base_rss, "rss_after_models_mb": rss_mb(),
                     "tflite_file_kb": {n: os.path.getsize(p) / 1024 for n, p in paths.items()},
                     "parameters": params}
    window = np.random.default_rng(42).random((1, SEQ, NFEAT)).astype(np.float32)

    def run_branch(name):
        it = interps[name]
        it.set_tensor(it.get_input_details()[0]["index"], window)
        it.invoke()
        return it.get_tensor(it.get_output_details()[0]["index"])

    def ensemble():
        ps = np.stack([run_branch(n)[0] for n in ("cnn", "bilstm", "transformer")])
        c = ps.max(axis=1)
        w = c / c.sum()
        return (w[:, None] * ps).sum(axis=0)

    res["load_levels"] = []
    cores = os.cpu_count() or 1
    for k in range(0, cores + 1):
        print(f"[{k + 1}/{cores + 1}] measuring with {k} background CPU load(s)...", flush=True)
        ctx = mp.get_context("spawn")
        stop = ctx.Event()
        procs = [ctx.Process(target=burner, args=(stop,), daemon=True) for _ in range(k)]
        for p in procs:
            p.start()
        time.sleep(0.5)
        lvl = {"background_cpu_burners": k}
        for name in ("cnn", "bilstm", "transformer"):
            lvl[name] = timed(lambda: run_branch(name), a.n, a.warmup)
        lvl["ensemble"] = timed(ensemble, a.n, a.warmup)
        stop.set()
        for p in procs:
            p.join()
        res["load_levels"].append(lvl)
        print(json.dumps({"k": k, "ensemble_p50": lvl["ensemble"]["p50_ms"], "ensemble_p95": lvl["ensemble"]["p95_ms"]}), flush=True)

    # sustained throughput + CPU utilisation + optional energy
    try:
        import psutil
        proc = psutil.Process()
        proc.cpu_percent(None)
    except ImportError:
        proc = None
    e0 = rapl_joules()
    print("\n>>> SUSTAINED PHASE STARTED: READ THE POWER METER NOW (30 seconds) <<<", flush=True)
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < 30.0:
        ensemble()
        n += 1
    dt = time.perf_counter() - t0
    e1 = rapl_joules()
    print(">>> SUSTAINED PHASE FINISHED <<<\n", flush=True)
    res["sustained"] = {"seconds": dt, "windows": n, "windows_per_s": n / dt,
                        "cpu_percent_of_one_core": proc.cpu_percent(None) if proc else None}
    if e0 is not None and e1 is not None:
        res["energy"] = {"source": "RAPL package counters", "joules_per_window": (e1 - e0) / n}
    elif a.meter_watts:
        res["energy"] = {"source": "external meter (user-entered average watts)",
                         "joules_per_window": a.meter_watts * dt / n}
    else:
        res["energy"] = {"source": "NOT MEASURED: no RAPL counters on this host and no external meter reading"}
    res["per_flow_cost_us_for_ns3"] = {
        "definition": "one ensemble window inference per arriving flow (stride-1 deployment); "
                      "idsLatencyUs = ensemble p50 at the given load level",
        "idle_p50_us": res["load_levels"][0]["ensemble"]["p50_ms"] * 1000,
        "idle_p95_us": res["load_levels"][0]["ensemble"]["p95_ms"] * 1000,
        "full_load_p50_us": res["load_levels"][-1]["ensemble"]["p50_ms"] * 1000,
        "full_load_p95_us": res["load_levels"][-1]["ensemble"]["p95_ms"] * 1000,
    }
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res["per_flow_cost_us_for_ns3"], indent=2))


if __name__ == "__main__":
    main()
