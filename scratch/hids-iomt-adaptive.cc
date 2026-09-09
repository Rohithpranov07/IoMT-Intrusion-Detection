/*
 * hids-iomt-adaptive.cc — network-level validation of the adaptive-threshold contribution.
 * NS3-Simulation.md tasks T-N1, T-N2, T-N4, T-N5.
 *
 * ============================================================================================
 * TOPOLOGY (NS3-Simulation.md §A.1 last rule, T-N1: these labels are unambiguous by construction)
 * ============================================================================================
 *   nWearables   legitimate IoMT patient devices. Distinct nodes. Periodic telemetry.
 *   nAttackers   attacker-bot nodes. DISTINCT NODES, never a subset of the wearables. The
 *                ambiguous "20 wearables / 60 compromised" phrasing NS3-Simulation.md §A.1 flags
 *                is resolved here in the only way that composes: 20 + 60 = 80 leaf senders.
 *   nFog         fog nodes running the IDS. Each leaf is pinned to fog node (index % nFog).
 *
 *   wearables ---\                      /--- fog[0]  (IDS)
 *   attackers ----+--- [edge switch] ---+--- fog[1]  (IDS)
 *                                        \-- ...
 *
 * ============================================================================================
 * THE IDS SERVICE MODEL, AND WHAT idsLatencyUs MEANS
 * ============================================================================================
 * Each fog node is a single-server queue. One flow costs `idsLatencyUs` of service time, so
 *     per-node capacity  = 1e6 / idsLatencyUs   flows/s
 *     aggregate capacity = nFog * per-node      flows/s
 * BOTH are printed. NS3-Simulation.md §B.2 flags that an earlier run printed a capacity equal to
 * 1/latency without saying which it was; printing both removes the ambiguity rather than
 * documenting it away.
 *
 * ============================================================================================
 * PROVENANCE OF idsLatencyUs  (NS3-Simulation.md §A.1 rule 1 and 2, T-N3)
 * ============================================================================================
 * This program NEVER supplies a default latency that claims to be a measurement. `--idsLatencyUs`
 * is a swept input and the summary labels it as such. At the time of writing:
 *   - `Build-Instructions.md` T4.3's real Raspberry Pi 4B measurement DOES NOT EXIST yet
 *     (`src/deployment/benchmark.py` refuses to produce one off the target hardware), so no run
 *     from this program may be labelled "this project's architecture on the Pi".
 *   - The base paper reports no per-flow inference latency at all (`PRD.md §2.2`), so no run may
 *     be labelled "the base paper's number" either.
 * See docs/latency_provenance.md. The `--latencySource` string is stamped into the output so a
 * saved run can never be read back without its provenance attached.
 *
 * ============================================================================================
 * THE THREE THRESHOLD RULES (T-N4)
 * ============================================================================================
 * All three decide the same question: is this source's inter-message gap too short? That is the
 * quantity `PRD.md §2.1.4` objects to the base paper fixing at a flat tau = 500 ms.
 *
 *  fixed        gap < 500 ms  ->  violation.  The base paper's rule (BASE_PAPER_FIXED_THRESHOLD_MS
 *               in src/adaptive/threshold.py). One constant for every device.
 *
 *  ewma         gap_ewma < tau(load)  ->  violation, where gap_ewma is a per-source EWMA
 *               (alpha = 0.30) of inter-message gaps and tau couples to the fog node's queue
 *               occupancy `load`. THIS RULE IS THIS PROJECT'S OWN RECONSTRUCTION of a
 *               congestion-adaptive detector, not a transcription of anyone's published code.
 *               Both couplings are selectable, because which one you pick decides which regime
 *               the detector goes blind in -- and that is the finding T-N2 exists to establish:
 *                 --ewmaCoupling=inverse : tau = 500 * (1 - load).  Detection loosens as the
 *                     queue fills, so a CONGESTED fog node stops alerting.
 *                 --ewmaCoupling=direct  : tau = 500 * load.  Detection only switches on once a
 *                     queue exists, so a FAST fog node never alerts. This is the mechanism
 *                     NS3-Simulation.md B.2 hypothesises for the C_batched run.
 *               See docs/zero_detection_investigation.md for the measured outcome of each.
 *
 *  criticality  gap < clamp(500 ms * criticality_weight * context_factor, 50, 2000) -> violation,
 *               with context_factor = 1 + 0.30 * (load - 0.50). This is `TRD.md §5.1`'s formula
 *               and it is kept numerically identical to src/adaptive/threshold.py's frozen
 *               constants (T3.4): weights 0.60 / 0.80 / 1.00 / 1.30, CONTEXT_SENSITIVITY 0.30,
 *               LOAD_NEUTRAL 0.50, floor 50 ms, ceiling 2000 ms.
 *
 * A source is blocked after `blockAfter` violations. Blocking a wearable is a FALSE POSITIVE and
 * is counted separately from blocking an attacker-bot — that separation is what `PRD.md §3.1`'s
 * >30% false-positive-reduction target is measured against.
 *
 * ============================================================================================
 * DETERMINISM (NS3-Simulation.md §A.3)
 * ============================================================================================
 * `--rngRun` sets the NS-3 run number; every stochastic element (telemetry jitter, start offsets)
 * draws from seeded streams. Two runs with the same arguments produce identical output.
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"

#include <algorithm>
#include <iomanip>
#include <map>
#include <queue>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("HidsIomtAdaptive");

// --- Constants mirrored EXACTLY from src/adaptive/threshold.py (T3.4). Do not drift. ---------
static const double kInterMessageBaseMs = 500.0;   // INTER_MESSAGE_BASE_MS
static const double kInterMessageFloorMs = 50.0;   // INTER_MESSAGE_FLOOR_MS
static const double kInterMessageCeilingMs = 2000.0; // INTER_MESSAGE_CEILING_MS
static const double kContextSensitivity = 0.30;    // CONTEXT_SENSITIVITY
static const double kLoadNeutral = 0.50;           // LOAD_NEUTRAL
static const double kEwmaAlpha = 0.30;             // this simulation's own choice, documented above

enum class RuleKind
{
    FIXED,
    EWMA,
    CRITICALITY
};

/** How the EWMA rule's threshold couples to fog-node queue occupancy. See the header comment. */
enum class EwmaCoupling
{
    INVERSE, //!< tau = base * (1 - load): blind when congested
    DIRECT   //!< tau = base * load:       blind when fast
};

/** Per-source detection state held by a fog node's IDS. */
struct SourceState
{
    double lastArrival = -1.0;   //!< seconds
    double ewmaGap = -1.0;       //!< seconds
    uint32_t violations = 0;
    uint32_t packets = 0;
    bool blocked = false;
    bool legitimate = true;      //!< false for attacker-bots
    double criticalityWeight = 1.0;
    double blockedAt = -1.0;
};

/**
 * Fog-node IDS: a single-server queue that inspects one flow per `idsLatencyUs` and applies one
 * of the three threshold rules to each inspected flow.
 */
class FogIds : public Application
{
  public:
    static TypeId GetTypeId();

    void Setup(uint16_t port,
               Time serviceTime,
               uint32_t queueCapacity,
               RuleKind rule,
               uint32_t blockAfter,
               EwmaCoupling coupling);

    /** Register a source's ground truth so alerts can be scored. Called at build time. */
    void RegisterSource(Ipv4Address address, bool legitimate, double criticalityWeight);

    /** Switch the rule's base threshold mid-run. T-N5's incremental-update stand-in. */
    void ApplyIncrementalUpdate(double newBaseMs);

    // --- Counters read by the summary ---
    uint64_t m_received = 0;      //!< packets that reached the fog node
    uint64_t m_queueDrops = 0;    //!< dropped because the IDS queue was full
    uint64_t m_blockedDrops = 0;  //!< dropped because the source was already blocked
    uint64_t m_inspected = 0;     //!< packets that completed IDS service
    uint64_t m_alerts = 0;
    uint64_t m_alertsOnAttackers = 0;
    uint64_t m_alertsOnWearables = 0;   //!< false-positive alerts
    uint64_t m_alertsBefore = 0;        //!< phase split for T-N5
    uint64_t m_alertsAfter = 0;
    uint64_t m_falseAlertsBefore = 0;
    uint64_t m_falseAlertsAfter = 0;
    std::map<Ipv4Address, SourceState> m_sources;

    /** Trace of (time, load, effective threshold ms, ewma ms) — the T-N2 instrumentation. */
    std::vector<std::array<double, 4>> m_trace;
    double m_phaseSplit = -1.0;
    bool m_traceEnabled = false;

  private:
    void StartApplication() override;
    void StopApplication() override;
    void HandleRead(Ptr<Socket> socket);
    void ServiceComplete();
    void Inspect(Ipv4Address source, double arrival);
    double CurrentLoad() const;
    double EffectiveThresholdMs(const SourceState& state) const;

    Ptr<Socket> m_socket;
    uint16_t m_port = 9;
    Time m_serviceTime;
    uint32_t m_queueCapacity = 200;
    RuleKind m_rule = RuleKind::FIXED;
    EwmaCoupling m_coupling = EwmaCoupling::INVERSE;
    uint32_t m_blockAfter = 3;
    double m_baseThresholdMs = kInterMessageBaseMs;
    std::queue<std::pair<Ipv4Address, double>> m_queue;
    bool m_busy = false;
    EventId m_serviceEvent;
};

TypeId
FogIds::GetTypeId()
{
    static TypeId tid =
        TypeId("FogIds").SetParent<Application>().SetGroupName("Applications").AddConstructor<FogIds>();
    return tid;
}

void
FogIds::Setup(uint16_t port,
              Time serviceTime,
              uint32_t queueCapacity,
              RuleKind rule,
              uint32_t blockAfter,
              EwmaCoupling coupling)
{
    m_port = port;
    m_serviceTime = serviceTime;
    m_queueCapacity = queueCapacity;
    m_rule = rule;
    m_blockAfter = blockAfter;
    m_coupling = coupling;
}

void
FogIds::RegisterSource(Ipv4Address address, bool legitimate, double criticalityWeight)
{
    SourceState state;
    state.legitimate = legitimate;
    state.criticalityWeight = criticalityWeight;
    m_sources[address] = state;
}

void
FogIds::ApplyIncrementalUpdate(double newBaseMs)
{
    m_baseThresholdMs = newBaseMs;
}

void
FogIds::StartApplication()
{
    m_socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
    m_socket->Bind(InetSocketAddress(Ipv4Address::GetAny(), m_port));
    m_socket->SetRecvCallback(MakeCallback(&FogIds::HandleRead, this));
}

void
FogIds::StopApplication()
{
    if (m_socket)
    {
        m_socket->Close();
    }
    Simulator::Cancel(m_serviceEvent);
}

double
FogIds::CurrentLoad() const
{
    return std::min(1.0, static_cast<double>(m_queue.size()) / static_cast<double>(m_queueCapacity));
}

/**
 * The three rules, side by side. Returns the inter-message gap, in ms, below which a flow counts
 * as a violation for this source right now.
 */
double
FogIds::EffectiveThresholdMs(const SourceState& state) const
{
    const double load = CurrentLoad();
    switch (m_rule)
    {
    case RuleKind::FIXED:
        // The base paper: one constant for every device, regardless of anything.
        return m_baseThresholdMs;

    case RuleKind::EWMA:
        // Congestion-adaptive. INVERSE goes blind at load 1 (threshold -> 0 ms, nothing can be
        // shorter); DIRECT goes blind at load 0 (same, at the other end). Neither is a bug: both
        // are what "let the queue state drive the threshold" means, and T-N2 measures both.
        return (m_coupling == EwmaCoupling::INVERSE) ? m_baseThresholdMs * (1.0 - load)
                                                     : m_baseThresholdMs * load;

    case RuleKind::CRITICALITY:
    default:
    {
        // TRD.md §5.1 / src/adaptive/threshold.py, constants identical.
        const double contextFactor = 1.0 + kContextSensitivity * (load - kLoadNeutral);
        const double raw = m_baseThresholdMs * state.criticalityWeight * contextFactor;
        return std::min(kInterMessageCeilingMs, std::max(kInterMessageFloorMs, raw));
    }
    }
}

void
FogIds::HandleRead(Ptr<Socket> socket)
{
    Ptr<Packet> packet;
    Address from;
    while ((packet = socket->RecvFrom(from)))
    {
        const Ipv4Address source = InetSocketAddress::ConvertFrom(from).GetIpv4();
        m_received++;

        auto it = m_sources.find(source);
        if (it != m_sources.end() && it->second.blocked)
        {
            // Already blocked: the fog node drops it without spending IDS service time on it.
            m_blockedDrops++;
            continue;
        }

        if (m_queue.size() >= m_queueCapacity)
        {
            m_queueDrops++;
            continue;
        }

        m_queue.push({source, Simulator::Now().GetSeconds()});
        if (!m_busy)
        {
            m_busy = true;
            m_serviceEvent = Simulator::Schedule(m_serviceTime, &FogIds::ServiceComplete, this);
        }
    }
}

void
FogIds::ServiceComplete()
{
    if (m_queue.empty())
    {
        m_busy = false;
        return;
    }
    const auto entry = m_queue.front();
    m_queue.pop();
    Inspect(entry.first, entry.second);
    m_inspected++;

    if (!m_queue.empty())
    {
        m_serviceEvent = Simulator::Schedule(m_serviceTime, &FogIds::ServiceComplete, this);
    }
    else
    {
        m_busy = false;
    }
}

void
FogIds::Inspect(Ipv4Address source, double arrival)
{
    auto it = m_sources.find(source);
    if (it == m_sources.end())
    {
        return; // unregistered source; cannot be scored, so it is not judged
    }
    SourceState& state = it->second;
    state.packets++;

    if (state.lastArrival < 0.0)
    {
        state.lastArrival = arrival;
        return; // no gap yet
    }

    const double gapMs = (arrival - state.lastArrival) * 1000.0;
    state.lastArrival = arrival;
    state.ewmaGap = (state.ewmaGap < 0.0) ? gapMs : kEwmaAlpha * gapMs + (1.0 - kEwmaAlpha) * state.ewmaGap;

    const double thresholdMs = EffectiveThresholdMs(state);
    const double observedMs = (m_rule == RuleKind::EWMA) ? state.ewmaGap : gapMs;

    if (m_traceEnabled)
    {
        m_trace.push_back({Simulator::Now().GetSeconds(), CurrentLoad(), thresholdMs, observedMs});
    }

    if (observedMs < thresholdMs)
    {
        state.violations++;
        m_alerts++;
        const bool afterSplit = (m_phaseSplit > 0.0 && Simulator::Now().GetSeconds() >= m_phaseSplit);
        if (afterSplit)
        {
            m_alertsAfter++;
        }
        else
        {
            m_alertsBefore++;
        }

        if (state.legitimate)
        {
            m_alertsOnWearables++; // false positive
            if (afterSplit)
            {
                m_falseAlertsAfter++;
            }
            else
            {
                m_falseAlertsBefore++;
            }
        }
        else
        {
            m_alertsOnAttackers++;
        }

        if (state.violations >= m_blockAfter && !state.blocked)
        {
            state.blocked = true;
            state.blockedAt = Simulator::Now().GetSeconds();
        }
    }
}

/**
 * Telemetry / flood sender with seeded jitter. UdpClient sends on a fixed interval, which would
 * make every legitimate gap identical and every false positive an all-or-nothing artefact of one
 * comparison; jitter is what makes the false-positive counts meaningful.
 */
class JitteredSender : public Application
{
  public:
    static TypeId GetTypeId();

    void Setup(Address peer, Time interval, double jitterFraction, uint32_t packetSize, Time startAt, Time stopAt);

  private:
    void StartApplication() override;
    void StopApplication() override;
    void SendOne();

    Ptr<Socket> m_socket;
    Address m_peer;
    Time m_interval;
    double m_jitter = 0.0;
    uint32_t m_packetSize = 64;
    Time m_startAt;
    Time m_stopAt;
    EventId m_sendEvent;
    Ptr<UniformRandomVariable> m_random;
    bool m_running = false;
};

TypeId
JitteredSender::GetTypeId()
{
    static TypeId tid = TypeId("JitteredSender")
                            .SetParent<Application>()
                            .SetGroupName("Applications")
                            .AddConstructor<JitteredSender>();
    return tid;
}

void
JitteredSender::Setup(Address peer,
                      Time interval,
                      double jitterFraction,
                      uint32_t packetSize,
                      Time startAt,
                      Time stopAt)
{
    m_peer = peer;
    m_interval = interval;
    m_jitter = jitterFraction;
    m_packetSize = packetSize;
    m_startAt = startAt;
    m_stopAt = stopAt;
    m_random = CreateObject<UniformRandomVariable>();
}

void
JitteredSender::StartApplication()
{
    m_socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
    m_socket->Connect(m_peer);
    m_running = true;
    const double offset = m_random->GetValue(0.0, m_interval.GetSeconds());
    m_sendEvent = Simulator::Schedule(m_startAt - Simulator::Now() + Seconds(offset),
                                      &JitteredSender::SendOne,
                                      this);
}

void
JitteredSender::StopApplication()
{
    m_running = false;
    Simulator::Cancel(m_sendEvent);
    if (m_socket)
    {
        m_socket->Close();
    }
}

void
JitteredSender::SendOne()
{
    if (!m_running || Simulator::Now() >= m_stopAt)
    {
        return;
    }
    m_socket->Send(Create<Packet>(m_packetSize));

    const double base = m_interval.GetSeconds();
    const double next = (m_jitter > 0.0) ? m_random->GetValue(base * (1.0 - m_jitter), base * (1.0 + m_jitter)) : base;
    m_sendEvent = Simulator::Schedule(Seconds(next), &JitteredSender::SendOne, this);
}

int
main(int argc, char* argv[])
{
    uint32_t nWearables = 20;
    uint32_t nAttackers = 60;
    uint32_t nFog = 4;
    double idsLatencyUs = 500.0;
    std::string latencySource = "UNLABELLED-SWEEP";
    std::string ruleName = "fixed";
    std::string couplingName = "inverse";
    double simTime = 60.0;
    uint32_t queueCapacity = 200;
    uint32_t blockAfter = 3;
    uint32_t rngRun = 1;
    double wearableIntervalMs = 600.0;
    double wearableJitter = 0.40;
    double attackerIntervalMs = 5.0;
    double novelAttackAt = -1.0;   // T-N5: a slow-and-low wave starts here
    double incrementalAt = -1.0;   // T-N5: the update stand-in fires here
    double incrementalBaseMs = 500.0;
    std::string label = "run";
    bool traceRule = false;

    CommandLine cmd(__FILE__);
    cmd.AddValue("nWearables", "legitimate IoMT devices", nWearables);
    cmd.AddValue("nAttackers", "attacker-bot nodes (distinct from wearables)", nAttackers);
    cmd.AddValue("nFog", "fog nodes running the IDS", nFog);
    cmd.AddValue("idsLatencyUs", "IDS inspection cost per flow, microseconds (SWEPT INPUT)", idsLatencyUs);
    cmd.AddValue("latencySource", "provenance string for idsLatencyUs; stamped into the output", latencySource);
    cmd.AddValue("rule", "threshold rule: fixed | ewma | criticality", ruleName);
    cmd.AddValue("ewmaCoupling", "ewma rule only: inverse (blind when congested) | direct (blind when fast)", couplingName);
    cmd.AddValue("simTime", "simulated seconds", simTime);
    cmd.AddValue("queueCapacity", "IDS queue depth per fog node", queueCapacity);
    cmd.AddValue("blockAfter", "violations before a source is blocked", blockAfter);
    cmd.AddValue("rngRun", "NS-3 run number (determinism)", rngRun);
    cmd.AddValue("wearableIntervalMs", "telemetry period", wearableIntervalMs);
    cmd.AddValue("wearableJitter", "fractional jitter on telemetry period", wearableJitter);
    cmd.AddValue("attackerIntervalMs", "attacker-bot send period", attackerIntervalMs);
    cmd.AddValue("novelAttackAt", "T-N5: seconds at which the slow-and-low wave starts (-1 off)", novelAttackAt);
    cmd.AddValue("incrementalAt", "T-N5: seconds at which the update stand-in fires (-1 off)", incrementalAt);
    cmd.AddValue("incrementalBaseMs", "T-N5: base threshold after the update", incrementalBaseMs);
    cmd.AddValue("label", "run label, echoed into the output", label);
    cmd.AddValue("traceRule", "log the rule's threshold vs observed gap (T-N2)", traceRule);
    cmd.Parse(argc, argv);

    RuleKind rule = RuleKind::FIXED;
    if (ruleName == "ewma")
    {
        rule = RuleKind::EWMA;
    }
    else if (ruleName == "criticality")
    {
        rule = RuleKind::CRITICALITY;
    }
    else if (ruleName != "fixed")
    {
        std::cerr << "unknown rule '" << ruleName << "'; expected fixed | ewma | criticality\n";
        return 1;
    }

    EwmaCoupling coupling = EwmaCoupling::INVERSE;
    if (couplingName == "direct")
    {
        coupling = EwmaCoupling::DIRECT;
    }
    else if (couplingName != "inverse")
    {
        std::cerr << "unknown ewmaCoupling '" << couplingName << "'; expected inverse | direct\n";
        return 1;
    }

    RngSeedManager::SetSeed(42);
    RngSeedManager::SetRun(rngRun);

    // --- Topology -----------------------------------------------------------------------------
    NodeContainer wearables;
    wearables.Create(nWearables);
    NodeContainer attackers;
    attackers.Create(nAttackers);
    NodeContainer fog;
    fog.Create(nFog);
    NodeContainer edge;
    edge.Create(1);

    InternetStackHelper stack;
    stack.Install(wearables);
    stack.Install(attackers);
    stack.Install(fog);
    stack.Install(edge);

    PointToPointHelper access;
    access.SetDeviceAttribute("DataRate", StringValue("10Mbps"));
    access.SetChannelAttribute("Delay", StringValue("2ms"));

    PointToPointHelper backhaul;
    backhaul.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    backhaul.SetChannelAttribute("Delay", StringValue("1ms"));

    Ipv4AddressHelper ipv4;
    std::vector<Ipv4Address> fogAddresses;
    uint32_t subnet = 1;

    // fog <-> edge
    for (uint32_t i = 0; i < nFog; ++i)
    {
        NetDeviceContainer devices = backhaul.Install(NodeContainer(edge.Get(0), fog.Get(i)));
        std::ostringstream net;
        net << "10." << (subnet / 254) << "." << (subnet % 254) << ".0";
        ipv4.SetBase(net.str().c_str(), "255.255.255.0");
        Ipv4InterfaceContainer interfaces = ipv4.Assign(devices);
        fogAddresses.push_back(interfaces.GetAddress(1));
        subnet++;
    }

    // leaves <-> edge. Wearables first, then attacker-bots.
    std::vector<Ipv4Address> wearableAddresses;
    std::vector<Ipv4Address> attackerAddresses;
    NodeContainer leaves;
    for (uint32_t i = 0; i < nWearables; ++i)
    {
        leaves.Add(wearables.Get(i));
    }
    for (uint32_t i = 0; i < nAttackers; ++i)
    {
        leaves.Add(attackers.Get(i));
    }

    for (uint32_t i = 0; i < leaves.GetN(); ++i)
    {
        NetDeviceContainer devices = access.Install(NodeContainer(edge.Get(0), leaves.Get(i)));
        std::ostringstream net;
        net << "10." << (subnet / 254) << "." << (subnet % 254) << ".0";
        ipv4.SetBase(net.str().c_str(), "255.255.255.0");
        Ipv4InterfaceContainer interfaces = ipv4.Assign(devices);
        if (i < nWearables)
        {
            wearableAddresses.push_back(interfaces.GetAddress(1));
        }
        else
        {
            attackerAddresses.push_back(interfaces.GetAddress(1));
        }
        subnet++;
    }

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // --- IDS applications ---------------------------------------------------------------------
    const uint16_t port = 9000;
    std::vector<Ptr<FogIds>> idsApps;
    for (uint32_t i = 0; i < nFog; ++i)
    {
        Ptr<FogIds> app = CreateObject<FogIds>();
        app->Setup(port, MicroSeconds(idsLatencyUs), queueCapacity, rule, blockAfter, coupling);
        app->m_traceEnabled = traceRule;
        app->m_phaseSplit = incrementalAt;
        fog.Get(i)->AddApplication(app);
        app->SetStartTime(Seconds(0.0));
        app->SetStopTime(Seconds(simTime));
        idsApps.push_back(app);
    }

    // Criticality classes, assigned deterministically round-robin over the wearables so the mix
    // is reproducible and every class is represented (weights from src/adaptive/threshold.py).
    const double weights[4] = {0.60, 0.80, 1.00, 1.30};
    const char* weightNames[4] = {"LIFE_CRITICAL", "HIGH", "STANDARD", "NON_CLINICAL"};

    for (uint32_t i = 0; i < nWearables; ++i)
    {
        const uint32_t fogIndex = i % nFog;
        idsApps[fogIndex]->RegisterSource(wearableAddresses[i], true, weights[i % 4]);
    }
    for (uint32_t i = 0; i < nAttackers; ++i)
    {
        const uint32_t fogIndex = (nWearables + i) % nFog;
        // An attacker-bot's traffic is addressed to a device; STANDARD is the neutral choice and
        // avoids flattering the criticality rule by handing attackers the most sensitive class.
        idsApps[fogIndex]->RegisterSource(attackerAddresses[i], false, 1.00);
    }

    // --- Traffic ------------------------------------------------------------------------------
    for (uint32_t i = 0; i < nWearables; ++i)
    {
        Ptr<JitteredSender> sender = CreateObject<JitteredSender>();
        sender->Setup(InetSocketAddress(fogAddresses[i % nFog], port),
                      MilliSeconds(wearableIntervalMs),
                      wearableJitter,
                      64,
                      Seconds(1.0),
                      Seconds(simTime));
        wearables.Get(i)->AddApplication(sender);
        sender->SetStartTime(Seconds(1.0));
        sender->SetStopTime(Seconds(simTime));
    }

    for (uint32_t i = 0; i < nAttackers; ++i)
    {
        const double startAt = (novelAttackAt > 0.0 && i % 2 == 1) ? novelAttackAt : 2.0;
        // The "novel" wave is slow-and-low: just under the base paper's flat 500 ms, which is
        // exactly the gap a flat rule is blind to and a tightened one is not.
        const double intervalMs = (novelAttackAt > 0.0 && i % 2 == 1) ? 450.0 : attackerIntervalMs;
        Ptr<JitteredSender> sender = CreateObject<JitteredSender>();
        sender->Setup(InetSocketAddress(fogAddresses[(nWearables + i) % nFog], port),
                      MilliSeconds(intervalMs),
                      0.05,
                      64,
                      Seconds(startAt),
                      Seconds(simTime));
        attackers.Get(i)->AddApplication(sender);
        sender->SetStartTime(Seconds(startAt));
        sender->SetStopTime(Seconds(simTime));
    }

    if (incrementalAt > 0.0)
    {
        for (auto& app : idsApps)
        {
            Simulator::Schedule(Seconds(incrementalAt), &FogIds::ApplyIncrementalUpdate, app, incrementalBaseMs);
        }
    }

    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    // --- Summary ------------------------------------------------------------------------------
    uint64_t received = 0, queueDrops = 0, blockedDrops = 0, inspected = 0;
    uint64_t alerts = 0, alertsAttackers = 0, alertsWearables = 0;
    uint64_t alertsBefore = 0, alertsAfter = 0, falseBefore = 0, falseAfter = 0;
    uint32_t blockedAttackers = 0, blockedWearables = 0;
    for (auto& app : idsApps)
    {
        received += app->m_received;
        queueDrops += app->m_queueDrops;
        blockedDrops += app->m_blockedDrops;
        inspected += app->m_inspected;
        alerts += app->m_alerts;
        alertsAttackers += app->m_alertsOnAttackers;
        alertsWearables += app->m_alertsOnWearables;
        alertsBefore += app->m_alertsBefore;
        alertsAfter += app->m_alertsAfter;
        falseBefore += app->m_falseAlertsBefore;
        falseAfter += app->m_falseAlertsAfter;
        for (auto& entry : app->m_sources)
        {
            if (entry.second.blocked)
            {
                entry.second.legitimate ? blockedWearables++ : blockedAttackers++;
            }
        }
    }

    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    double delaySum = 0.0, jitterSum = 0.0;
    uint64_t rxPackets = 0, txPackets = 0, rxBytes = 0;
    for (auto& kv : monitor->GetFlowStats())
    {
        txPackets += kv.second.txPackets;
        rxPackets += kv.second.rxPackets;
        rxBytes += kv.second.rxBytes;
        delaySum += kv.second.delaySum.GetSeconds();
        jitterSum += kv.second.jitterSum.GetSeconds();
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n================ HIDS-IoMT NS-3 run: " << label << " ================\n";
    std::cout << "ALL NUMBERS BELOW ARE NS-3 SIMULATION OUTPUT, NOT PHYSICAL MEASUREMENTS.\n";
    std::cout << "  (NS3-Simulation.md A.1 rule 4: a simulated delay is not a Raspberry Pi number.)\n\n";

    std::cout << "-- topology (T-N1: roles are distinct node sets, never subsets) --\n";
    std::cout << "  wearables (legitimate) : " << nWearables << "\n";
    std::cout << "  attacker-bots          : " << nAttackers << "\n";
    std::cout << "  fog nodes (IDS)        : " << nFog << "\n";
    std::cout << "  total leaf senders     : " << (nWearables + nAttackers) << "\n\n";

    std::cout << "-- IDS service model --\n";
    std::cout << "  inspection cost/flow   : " << idsLatencyUs << " us\n";
    std::cout << "  latency provenance     : " << latencySource << "\n";
    std::cout << "  capacity PER FOG NODE  : " << (1e6 / idsLatencyUs) << " flows/s\n";
    std::cout << "  capacity AGGREGATE     : " << (nFog * 1e6 / idsLatencyUs) << " flows/s ("
              << nFog << " nodes)\n";
    std::cout << "  queue depth per node   : " << queueCapacity << "\n\n";

    std::cout << "-- threshold rule --\n";
    std::cout << "  rule                   : " << ruleName << "\n";
    if (rule == RuleKind::EWMA)
    {
        std::cout << "  ewma coupling          : " << couplingName
                  << (coupling == EwmaCoupling::INVERSE ? "  (tau = base * (1 - load))"
                                                        : "  (tau = base * load)")
                  << "\n";
    }
    std::cout << "  base threshold         : " << kInterMessageBaseMs << " ms\n";
    std::cout << "  block after            : " << blockAfter << " violations\n";
    if (rule == RuleKind::CRITICALITY)
    {
        std::cout << "  criticality weights    : ";
        for (int i = 0; i < 4; ++i)
        {
            std::cout << weightNames[i] << "=" << weights[i] << (i < 3 ? ", " : "\n");
        }
    }
    std::cout << "\n";

    std::cout << "-- detection --\n";
    std::cout << "  packets reaching fog   : " << received << "\n";
    std::cout << "  inspected by IDS       : " << inspected << "\n";
    std::cout << "  dropped, queue full    : " << queueDrops << "\n";
    std::cout << "  dropped, source blocked: " << blockedDrops << "\n";
    std::cout << "  alerts raised          : " << alerts << "\n";
    std::cout << "  ... on attacker-bots   : " << alertsAttackers << "  (true positives)\n";
    std::cout << "  ... on wearables       : " << alertsWearables << "  (FALSE POSITIVES)\n";
    std::cout << "  IDS drop ratio         : "
              << (received ? 100.0 * (queueDrops + blockedDrops) / received : 0.0)
              << " % of packets reaching the fog node\n";
    std::cout << "    (NOT the network loss ratio below: these packets arrived, then the IDS\n"
                 "     queue overflowed or the source was already blocked. Reporting one as the\n"
                 "     other is the conflation NS3-Simulation.md A.1 rule 4 forbids.)\n";
    std::cout << "  attacker-bots blocked  : " << blockedAttackers << " / " << nAttackers << "\n";
    std::cout << "  wearables blocked      : " << blockedWearables << " / " << nWearables
              << "  (FALSE POSITIVES)\n\n";

    if (incrementalAt > 0.0)
    {
        std::cout << "-- T-N5 phases (split at " << incrementalAt << "s) --\n";
        std::cout << "  alerts before          : " << alertsBefore << "  (false: " << falseBefore << ")\n";
        std::cout << "  alerts after           : " << alertsAfter << "  (false: " << falseAfter << ")\n\n";
    }

    std::cout << "-- network (FlowMonitor, simulated) --\n";
    std::cout << "  tx packets             : " << txPackets << "\n";
    std::cout << "  rx packets             : " << rxPackets << "\n";
    std::cout << "  delivery ratio         : "
              << (txPackets ? 100.0 * rxPackets / txPackets : 0.0) << " %\n";
    std::cout << "  loss ratio (network)   : "
              << (txPackets ? 100.0 * (txPackets - rxPackets) / txPackets : 0.0) << " %\n";
    std::cout << "  mean end-to-end delay  : " << (rxPackets ? 1000.0 * delaySum / rxPackets : 0.0)
              << " ms (SIMULATED)\n";
    std::cout << "  mean jitter            : " << (rxPackets ? 1000.0 * jitterSum / rxPackets : 0.0)
              << " ms (SIMULATED)\n";
    std::cout << "  throughput             : " << (8.0 * rxBytes / simTime / 1e6) << " Mbps\n";
    std::cout << "  rngRun                 : " << rngRun << " (seed 42)\n";

    if (traceRule)
    {
        std::cout << "\n-- T-N2 rule trace (time_s,load,threshold_ms,observed_ms), first 20 --\n";
        uint32_t shown = 0;
        for (auto& app : idsApps)
        {
            for (auto& row : app->m_trace)
            {
                if (shown++ >= 20)
                {
                    break;
                }
                std::cout << "  TRACE," << row[0] << "," << row[1] << "," << row[2] << "," << row[3] << "\n";
            }
            if (shown >= 20)
            {
                break;
            }
        }
        double maxLoad = 0.0, minThreshold = 1e9;
        uint64_t traceRows = 0;
        for (auto& app : idsApps)
        {
            for (auto& row : app->m_trace)
            {
                maxLoad = std::max(maxLoad, row[1]);
                minThreshold = std::min(minThreshold, row[2]);
                traceRows++;
            }
        }
        std::cout << "  TRACE_SUMMARY rows=" << traceRows << " max_load=" << maxLoad
                  << " min_threshold_ms=" << (traceRows ? minThreshold : 0.0) << "\n";
    }
    std::cout << "================ end of run: " << label << " ================\n";

    Simulator::Destroy();
    return 0;
}
