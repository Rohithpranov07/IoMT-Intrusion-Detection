# NS-3 test report

`scratch/hids-iomt-adaptive.cc`, seed 42, run 1. Real simulation output, reproducible with the
command at the bottom. Two field names are adapted from your reference format to what this
program actually computes; both are marked inline, nothing is invented.

```
wearables / attacker-bots / fog : 20 / 5 / 4
CNN-LSTM latency                : 691.9 us/flow   (swept value; no source cited)
IDS                              : on   (threshold rule = fixed, tau = 500 ms)
tau                               : 500 ms
ready interval                    : n/a — this program inspects per-flow continuously,
                                      it has no periodic 30 s check cycle to report

NETWORK (what the paper never measures)
mean end-to-end delay     : 3.0827 ms
mean jitter                : 0.0000 ms
throughput                 : 0.7106 Mbps
delivery ratio               : 99.9896 %
packet loss                  : 0.0104 %

HIDS-IoMT
messages accepted         : 276     (packets the IDS actually serviced)
messages rejected          : 28689   (dropped: queue full or source already blocked)
alerts raised               : 75
nodes blocked                : 5 of 5 attacker-bots, 20 of 20 wearables (false positives)
```

## Reproducing

```bash
cd ~/ns-3-dev
./ns3 run "hids-iomt-adaptive --rule=fixed --nWearables=20 --nAttackers=5 --nFog=4 \
  --idsLatencyUs=691.9 --simTime=30 --label=output_md_run \
  --latencySource=USER-REQUESTED-691.9us-no-source-given"
```

## Reading this

- **5 of 5 attackers blocked.** A flat 500 ms threshold reliably catches a 5 ms flood.
- **20 of 20 wearables also blocked** — every one is a false positive. A flat threshold has
  no per-device criticality, so it quarantines the whole fleet, not just the attackers. This
  project's own `criticality` rule exists to fix that; it was not selected in this run.
- **messages accepted/rejected** has no equivalent field in this codebase, so it is defined
  here as: accepted = packets the IDS queue actually serviced (276); rejected = packets
  dropped before service, either because the queue was full or because the source was already
  blocked (28,689). Nearly all rejections are the second kind, which is the system doing its
  job (a blocked bot keeps flooding, keeps being refused) rather than a capacity failure.
