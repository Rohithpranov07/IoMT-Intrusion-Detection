/*
 * hids-iomt-r3.cc : Review 3 extension of hids-iomt-adaptive.cc
 *
 * With every new option at its default, this program is the Review 2 simulator: same topology,
 * same node creation order, same random-stream order, same packet size, same three rules. The
 * regression check in scripts/regression_check.py re-runs the Review 2 head-to-head and requires
 * the published counts (A 327/60/20, B 303/36/8, C 296/29/9) to come back unchanged.
 *
 * New in Review 3 (all OFF by default):
 *   Phase 1  --baseMs --ewmaAlpha --weightScheme --deviceMix --topology   (scenario + sensitivity)
 *            application-level delay (includes IDS queueing, which FlowMonitor never saw),
 *            wearable PDR, false negatives, control overhead, IDS energy, time-to-block
 *   Phase 3  --mitigation=hysteresis|dual|capacity   (latch-off mitigations for the EWMA rule)
 *   Phase 4  --slowFraction --slowMinMs --slowMaxMs   (attackers that overlap legitimate timing)
 *            --twoStage --stage2Tpr --stage2Fpr --stage2Mode   (threshold AND classifier)
 *   Phase 5  --idsLatencyUs stays the swept input; --latencySource carries provenance
 *
 * New random draws (topology placement, slow-attacker periods, stage-2 flags) use a separate
 * std::mt19937 seeded from rngRun, so they never perturb ns-3's own stream assignment.
 *
 * Every run ends with ONE machine-readable line beginning "RESULT," of key=value pairs.
 * ALL NUMBERS ARE NS-3 SIMULATION OUTPUT, NOT PHYSICAL MEASUREMENTS.
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("HidsIomtR3");

static const double kInterMessageFloorMs = 50.0;
static const double kInterMessageCeilingMs = 2000.0;
static const double kContextSensitivity = 0.30;
static const double kLoadNeutral = 0.50;
static const uint32_t kSeqTsBytes = 12; // SeqTsHeader size; payload is reduced to keep 64 B total

enum class RuleKind { FIXED, EWMA, CRITICALITY };
enum class EwmaCoupling { INVERSE, DIRECT };
enum class Mitigation { NONE, HYSTERESIS, DUAL, CAPACITY };

struct SourceState
{
    double lastArrival = -1.0;
    double ewmaGap = -1.0;
    uint32_t violations = 0;
    uint32_t packets = 0;
    bool blocked = false;
    bool legitimate = true;
    double criticalityWeight = 1.0;
    double blockedAt = -1.0;
    bool stage2Anomalous = false; // used by stage2Mode=correlated
};

struct Stage2Config
{
    bool enabled = false;
    double tpr = 1.0;
    double fpr = 0.0;
    bool correlated = false; // false: independent per packet; true: one verdict per source
};

struct MitigationConfig
{
    Mitigation kind = Mitigation::NONE;
    double hysFloor = 0.30;  // floor as a fraction of base
    double hysHigh = 0.80;   // enter saturated mode above this load
    double hysLow = 0.20;    // leave saturated mode below this load
    double dualHardMs = 100.0;
    double capKnee = 0.80;   // utilisation above which the capacity rule stops relaxing
    double capWindowS = 1.0;
};

class FogIds : public Application
{
  public:
    static TypeId GetTypeId();
    void Setup(uint16_t port, Time serviceTime, uint32_t queueCapacity, RuleKind rule,
               uint32_t blockAfter, EwmaCoupling coupling, double baseMs, double alpha,
               MitigationConfig mit, Stage2Config s2, uint32_t rngSeed);
    void RegisterSource(Ipv4Address address, bool legitimate, double criticalityWeight);
    void ApplyIncrementalUpdate(double newBaseMs) { m_baseThresholdMs = newBaseMs; }
    void SetControlPeers(std::vector<InetSocketAddress> peers) { m_ctrlPeers = peers; }

    uint64_t m_received = 0, m_queueDrops = 0, m_blockedDrops = 0, m_inspected = 0;
    uint64_t m_alerts = 0, m_alertsOnAttackers = 0, m_alertsOnWearables = 0;
    uint64_t m_alertsBefore = 0, m_alertsAfter = 0, m_falseAlertsBefore = 0, m_falseAlertsAfter = 0;
    uint64_t m_legitDelivered = 0;
    double m_legitDelaySum = 0.0;
    std::vector<double> m_legitDelays;
    uint64_t m_ctrlTxPackets = 0, m_ctrlTxBytes = 0;
    double m_maxLoad = 0.0;
    double m_minThresholdMs = 1e9;
    std::map<Ipv4Address, SourceState> m_sources;
    double m_phaseSplit = -1.0;

  private:
    void StartApplication() override;
    void StopApplication() override;
    void HandleRead(Ptr<Socket> socket);
    void ServiceComplete();
    void Inspect(Ipv4Address source, double arrival, double sendTs);
    double CurrentLoad() const;
    double Utilisation();
    double EffectiveThresholdMs(const SourceState& state);
    void SendControl(Ipv4Address blockedSource);

    Ptr<Socket> m_socket;
    Ptr<Socket> m_ctrlSocket;
    std::vector<InetSocketAddress> m_ctrlPeers;
    uint16_t m_port = 9;
    Time m_serviceTime;
    uint32_t m_queueCapacity = 200;
    RuleKind m_rule = RuleKind::FIXED;
    EwmaCoupling m_coupling = EwmaCoupling::INVERSE;
    uint32_t m_blockAfter = 3;
    double m_baseThresholdMs = 500.0;
    double m_alpha = 0.30;
    MitigationConfig m_mit;
    Stage2Config m_s2;
    bool m_saturated = false;
    std::deque<double> m_arrivalTimes; // for the capacity rule's offered-rate estimate
    std::queue<std::array<double, 2>> m_queueTimes;
    std::queue<Ipv4Address> m_queueSrc;
    bool m_busy = false;
    EventId m_serviceEvent;
    std::mt19937 m_rng;
    std::uniform_real_distribution<double> m_unit{0.0, 1.0};
};

TypeId
FogIds::GetTypeId()
{
    static TypeId tid = TypeId("FogIdsR3").SetParent<Application>().SetGroupName("Applications").AddConstructor<FogIds>();
    return tid;
}

void
FogIds::Setup(uint16_t port, Time serviceTime, uint32_t queueCapacity, RuleKind rule,
              uint32_t blockAfter, EwmaCoupling coupling, double baseMs, double alpha,
              MitigationConfig mit, Stage2Config s2, uint32_t rngSeed)
{
    m_port = port;
    m_serviceTime = serviceTime;
    m_queueCapacity = queueCapacity;
    m_rule = rule;
    m_blockAfter = blockAfter;
    m_coupling = coupling;
    m_baseThresholdMs = baseMs;
    m_alpha = alpha;
    m_mit = mit;
    m_s2 = s2;
    m_rng.seed(rngSeed);
}

void
FogIds::RegisterSource(Ipv4Address address, bool legitimate, double criticalityWeight)
{
    SourceState state;
    state.legitimate = legitimate;
    state.criticalityWeight = criticalityWeight;
    if (m_s2.enabled && m_s2.correlated)
    {
        state.stage2Anomalous = m_unit(m_rng) < (legitimate ? m_s2.fpr : m_s2.tpr);
    }
    m_sources[address] = state;
}

void
FogIds::StartApplication()
{
    m_socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
    m_socket->Bind(InetSocketAddress(Ipv4Address::GetAny(), m_port));
    m_socket->SetRecvCallback(MakeCallback(&FogIds::HandleRead, this));
    m_ctrlSocket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
    m_ctrlSocket->Bind();
}

void
FogIds::StopApplication()
{
    if (m_socket) m_socket->Close();
    if (m_ctrlSocket) m_ctrlSocket->Close();
    Simulator::Cancel(m_serviceEvent);
}

double
FogIds::CurrentLoad() const
{
    return std::min(1.0, static_cast<double>(m_queueSrc.size()) / static_cast<double>(m_queueCapacity));
}

/** Offered utilisation rho = arrival rate x service time over a sliding window (capacity rule). */
double
FogIds::Utilisation()
{
    const double now = Simulator::Now().GetSeconds();
    while (!m_arrivalTimes.empty() && m_arrivalTimes.front() < now - m_mit.capWindowS)
    {
        m_arrivalTimes.pop_front();
    }
    const double rate = m_arrivalTimes.size() / m_mit.capWindowS;
    return rate * m_serviceTime.GetSeconds();
}

double
FogIds::EffectiveThresholdMs(const SourceState& state)
{
    const double load = CurrentLoad();
    switch (m_rule)
    {
    case RuleKind::FIXED:
        return m_baseThresholdMs;
    case RuleKind::EWMA:
    {
        double tau = (m_coupling == EwmaCoupling::INVERSE) ? m_baseThresholdMs * (1.0 - load)
                                                           : m_baseThresholdMs * load;
        if (m_mit.kind == Mitigation::HYSTERESIS)
        {
            // Option B: a band with memory. Above hysHigh the rule enters a saturated mode and
            // holds its threshold at the floor until load falls below hysLow; it can never go
            // below the floor, so the threshold cannot collapse to zero.
            if (!m_saturated && load >= m_mit.hysHigh) m_saturated = true;
            if (m_saturated && load <= m_mit.hysLow) m_saturated = false;
            const double floorMs = m_mit.hysFloor * m_baseThresholdMs;
            tau = m_saturated ? floorMs : std::max(floorMs, tau);
        }
        else if (m_mit.kind == Mitigation::CAPACITY)
        {
            // Option A: couple to a measured capacity boundary, not to observed congestion.
            // Below the knee the rule relaxes with utilisation exactly as B intends. At or above
            // it, offered load exceeding what the fog node can serve is itself evidence of a
            // flood, so the rule stops relaxing and returns to the strict base threshold.
            const double rho = Utilisation();
            tau = (rho < m_mit.capKnee) ? m_baseThresholdMs * (1.0 - rho) : m_baseThresholdMs;
        }
        return tau;
    }
    case RuleKind::CRITICALITY:
    default:
    {
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
        if (m_mit.kind == Mitigation::CAPACITY)
        {
            m_arrivalTimes.push_back(Simulator::Now().GetSeconds());
        }
        SeqTsHeader ts;
        double sendTs = -1.0;
        if (packet->GetSize() >= kSeqTsBytes)
        {
            packet->RemoveHeader(ts);
            sendTs = ts.GetTs().GetSeconds();
        }
        auto it = m_sources.find(source);
        if (it != m_sources.end() && it->second.blocked)
        {
            m_blockedDrops++;
            continue;
        }
        if (m_queueSrc.size() >= m_queueCapacity)
        {
            m_queueDrops++;
            continue;
        }
        m_queueSrc.push(source);
        m_queueTimes.push({Simulator::Now().GetSeconds(), sendTs});
        m_maxLoad = std::max(m_maxLoad, CurrentLoad());
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
    if (m_queueSrc.empty())
    {
        m_busy = false;
        return;
    }
    const Ipv4Address src = m_queueSrc.front();
    const auto times = m_queueTimes.front();
    m_queueSrc.pop();
    m_queueTimes.pop();
    Inspect(src, times[0], times[1]);
    m_inspected++;
    if (!m_queueSrc.empty())
    {
        m_serviceEvent = Simulator::Schedule(m_serviceTime, &FogIds::ServiceComplete, this);
    }
    else
    {
        m_busy = false;
    }
}

void
FogIds::SendControl(Ipv4Address blockedSource)
{
    // Quarantine notification: one 48-byte message to every peer fog node and the controller.
    for (auto& peer : m_ctrlPeers)
    {
        Ptr<Packet> p = Create<Packet>(48);
        m_ctrlSocket->SendTo(p, 0, peer);
        m_ctrlTxPackets++;
        m_ctrlTxBytes += 48;
    }
}

void
FogIds::Inspect(Ipv4Address source, double arrival, double sendTs)
{
    auto it = m_sources.find(source);
    if (it == m_sources.end())
    {
        return;
    }
    SourceState& state = it->second;
    state.packets++;

    // A packet that completes inspection from a not-yet-blocked wearable is delivered.
    auto deliver = [&]() {
        if (state.legitimate && !state.blocked && sendTs >= 0.0)
        {
            const double d = Simulator::Now().GetSeconds() - sendTs;
            m_legitDelivered++;
            m_legitDelaySum += d;
            if (m_legitDelays.size() < 200000) m_legitDelays.push_back(d);
        }
    };

    if (state.lastArrival < 0.0)
    {
        state.lastArrival = arrival;
        deliver();
        return;
    }

    const double gapMs = (arrival - state.lastArrival) * 1000.0;
    state.lastArrival = arrival;
    state.ewmaGap = (state.ewmaGap < 0.0) ? gapMs : m_alpha * gapMs + (1.0 - m_alpha) * state.ewmaGap;

    const double thresholdMs = EffectiveThresholdMs(state);
    m_minThresholdMs = std::min(m_minThresholdMs, thresholdMs);
    const double observedMs = (m_rule == RuleKind::EWMA) ? state.ewmaGap : gapMs;

    bool violation = observedMs < thresholdMs;
    if (m_rule == RuleKind::EWMA && m_mit.kind == Mitigation::DUAL)
    {
        // Option C: a fixed hard threshold on the raw gap, OR the adaptive one.
        violation = violation || (gapMs < m_mit.dualHardMs);
    }
    if (violation && m_s2.enabled)
    {
        // Stage 2: the flow-level classifier must agree before the violation counts.
        bool flag;
        if (m_s2.correlated)
        {
            flag = state.stage2Anomalous;
        }
        else
        {
            flag = m_unit(m_rng) < (state.legitimate ? m_s2.fpr : m_s2.tpr);
        }
        violation = flag;
    }

    if (violation)
    {
        state.violations++;
        m_alerts++;
        const bool afterSplit = (m_phaseSplit > 0.0 && Simulator::Now().GetSeconds() >= m_phaseSplit);
        afterSplit ? m_alertsAfter++ : m_alertsBefore++;
        if (state.legitimate)
        {
            m_alertsOnWearables++;
            afterSplit ? m_falseAlertsAfter++ : m_falseAlertsBefore++;
        }
        else
        {
            m_alertsOnAttackers++;
        }
        if (state.violations >= m_blockAfter && !state.blocked)
        {
            state.blocked = true;
            state.blockedAt = Simulator::Now().GetSeconds();
            SendControl(source);
        }
    }
    deliver();
}

class CtrlSink : public Application
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("CtrlSinkR3").SetParent<Application>().AddConstructor<CtrlSink>();
        return tid;
    }
    uint64_t m_rx = 0;
    uint16_t m_port = 9100;

  private:
    void StartApplication() override
    {
        m_socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
        m_socket->Bind(InetSocketAddress(Ipv4Address::GetAny(), m_port));
        m_socket->SetRecvCallback(MakeCallback(&CtrlSink::HandleRead, this));
    }
    void StopApplication() override
    {
        if (m_socket) m_socket->Close();
    }
    void HandleRead(Ptr<Socket> s)
    {
        Ptr<Packet> p;
        while ((p = s->Recv())) m_rx++;
    }
    Ptr<Socket> m_socket;
};

class JitteredSender : public Application
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("JitteredSenderR3").SetParent<Application>().AddConstructor<JitteredSender>();
        return tid;
    }
    void Setup(Address peer, Time interval, double jitterFraction, uint32_t packetSize, Time startAt, Time stopAt)
    {
        m_peer = peer;
        m_interval = interval;
        m_jitter = jitterFraction;
        m_packetSize = packetSize;
        m_startAt = startAt;
        m_stopAt = stopAt;
        m_random = CreateObject<UniformRandomVariable>();
    }
    uint64_t m_sent = 0;

  private:
    void StartApplication() override
    {
        m_socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
        m_socket->Connect(m_peer);
        m_running = true;
        const double offset = m_random->GetValue(0.0, m_interval.GetSeconds());
        m_sendEvent = Simulator::Schedule(m_startAt - Simulator::Now() + Seconds(offset), &JitteredSender::SendOne, this);
    }
    void StopApplication() override
    {
        m_running = false;
        Simulator::Cancel(m_sendEvent);
        if (m_socket) m_socket->Close();
    }
    void SendOne()
    {
        if (!m_running || Simulator::Now() >= m_stopAt) return;
        Ptr<Packet> p = Create<Packet>(m_packetSize - kSeqTsBytes);
        SeqTsHeader ts;
        ts.SetSeq(static_cast<uint32_t>(m_sent));
        p->AddHeader(ts);
        m_socket->Send(p);
        m_sent++;
        const double base = m_interval.GetSeconds();
        const double next = (m_jitter > 0.0) ? m_random->GetValue(base * (1.0 - m_jitter), base * (1.0 + m_jitter)) : base;
        m_sendEvent = Simulator::Schedule(Seconds(next), &JitteredSender::SendOne, this);
    }
    Ptr<Socket> m_socket;
    Address m_peer;
    Time m_interval;
    double m_jitter = 0.0;
    uint32_t m_packetSize = 64;
    Time m_startAt, m_stopAt;
    EventId m_sendEvent;
    Ptr<UniformRandomVariable> m_random;
    bool m_running = false;
};

// ----------------------------------------------------------------------------------------------
// Device-class assignment. Balanced reproduces Review 2's round-robin exactly.
static std::vector<int>
AssignClasses(uint32_t n, const std::string& mix)
{
    std::vector<int> cls(n);
    if (mix == "balanced")
    {
        for (uint32_t i = 0; i < n; ++i) cls[i] = i % 4;
        return cls;
    }
    std::array<double, 4> p = (mix == "icu") ? std::array<double, 4>{0.50, 0.30, 0.15, 0.05}
                                             : std::array<double, 4>{0.05, 0.15, 0.30, 0.50};
    // Largest-remainder quota, then a deterministic interleave so classes spread across fog nodes.
    std::array<int, 4> q{};
    std::vector<std::pair<double, int>> rem;
    int used = 0;
    for (int k = 0; k < 4; ++k)
    {
        q[k] = static_cast<int>(std::floor(p[k] * n));
        used += q[k];
        rem.push_back({p[k] * n - q[k], k});
    }
    std::sort(rem.begin(), rem.end(), [](auto& a, auto& b) { return a.first > b.first; });
    for (int i = 0; used < static_cast<int>(n); ++i, ++used) q[rem[i % 4].second]++;
    uint32_t idx = 0;
    std::array<int, 4> left = q;
    while (idx < n)
    {
        for (int k = 0; k < 4 && idx < n; ++k)
        {
            if (left[k] > 0)
            {
                cls[idx++] = k;
                left[k]--;
            }
        }
    }
    return cls;
}

static std::array<double, 4>
Weights(const std::string& scheme)
{
    if (scheme == "uniform") return {1.0, 1.0, 1.0, 1.0};
    if (scheme == "linear") return {0.70, 0.90, 1.10, 1.30};
    if (scheme == "exponential") return {0.25, 0.50, 1.00, 2.00};
    return {0.60, 0.80, 1.00, 1.30}; // "frozen": src/adaptive/threshold.py (T3.4)
}

int
main(int argc, char* argv[])
{
    uint32_t nWearables = 20, nAttackers = 60, nFog = 4;
    double idsLatencyUs = 500.0;
    std::string latencySource = "UNLABELLED-SWEEP";
    std::string ruleName = "fixed", couplingName = "inverse";
    double simTime = 60.0;
    uint32_t queueCapacity = 200, blockAfter = 3, rngRun = 1;
    double wearableIntervalMs = 600.0, wearableJitter = 0.40, attackerIntervalMs = 5.0;
    double novelAttackAt = -1.0, incrementalAt = -1.0, incrementalBaseMs = 500.0;
    std::string label = "run";
    // Review 3
    double baseMs = 500.0, ewmaAlpha = 0.30;
    std::string weightScheme = "frozen", deviceMix = "balanced", topology = "star";
    std::string mitigationName = "none";
    MitigationConfig mit;
    double slowFraction = 0.0, slowMinMs = 300.0, slowMaxMs = 600.0, slowJitter = 0.40;
    bool twoStage = false;
    double stage2Tpr = 1.0, stage2Fpr = 0.0;
    std::string stage2Mode = "independent";
    double fogActiveW = 6.4, fogIdleW = 2.7;
    bool useFlowmon = true;

    CommandLine cmd(__FILE__);
    cmd.AddValue("nWearables", "legitimate IoMT devices", nWearables);
    cmd.AddValue("nAttackers", "attacker-bot nodes", nAttackers);
    cmd.AddValue("nFog", "fog nodes running the IDS", nFog);
    cmd.AddValue("idsLatencyUs", "IDS inspection cost per flow (us)", idsLatencyUs);
    cmd.AddValue("latencySource", "provenance of idsLatencyUs", latencySource);
    cmd.AddValue("rule", "fixed | ewma | criticality", ruleName);
    cmd.AddValue("ewmaCoupling", "inverse | direct", couplingName);
    cmd.AddValue("simTime", "simulated seconds", simTime);
    cmd.AddValue("queueCapacity", "IDS queue depth per fog node", queueCapacity);
    cmd.AddValue("blockAfter", "violations before blocking (patience)", blockAfter);
    cmd.AddValue("rngRun", "ns-3 run number", rngRun);
    cmd.AddValue("wearableIntervalMs", "telemetry period", wearableIntervalMs);
    cmd.AddValue("wearableJitter", "telemetry jitter fraction", wearableJitter);
    cmd.AddValue("attackerIntervalMs", "flood period", attackerIntervalMs);
    cmd.AddValue("novelAttackAt", "slow-and-low wave start (-1 off)", novelAttackAt);
    cmd.AddValue("incrementalAt", "update stand-in time (-1 off)", incrementalAt);
    cmd.AddValue("incrementalBaseMs", "base threshold after update", incrementalBaseMs);
    cmd.AddValue("label", "run label", label);
    cmd.AddValue("baseMs", "base threshold tau0 (ms)", baseMs);
    cmd.AddValue("ewmaAlpha", "EWMA smoothing alpha", ewmaAlpha);
    cmd.AddValue("weightScheme", "frozen | uniform | linear | exponential", weightScheme);
    cmd.AddValue("deviceMix", "balanced | icu | periphery", deviceMix);
    cmd.AddValue("topology", "star | clustered | grid | random", topology);
    cmd.AddValue("mitigation", "none | hysteresis | dual | capacity (EWMA rule)", mitigationName);
    cmd.AddValue("hysFloor", "hysteresis floor as fraction of base", mit.hysFloor);
    cmd.AddValue("hysHigh", "hysteresis enter load", mit.hysHigh);
    cmd.AddValue("hysLow", "hysteresis exit load", mit.hysLow);
    cmd.AddValue("dualHardMs", "dual: hard fixed threshold on raw gap (ms)", mit.dualHardMs);
    cmd.AddValue("capKnee", "capacity: utilisation knee", mit.capKnee);
    cmd.AddValue("capWindowS", "capacity: offered-rate window (s)", mit.capWindowS);
    cmd.AddValue("slowFraction", "fraction of attackers that are slow mimics from the start", slowFraction);
    cmd.AddValue("slowMinMs", "slow mimic period min", slowMinMs);
    cmd.AddValue("slowMaxMs", "slow mimic period max", slowMaxMs);
    cmd.AddValue("slowJitter", "slow mimic jitter", slowJitter);
    cmd.AddValue("twoStage", "require a stage-2 classifier flag", twoStage);
    cmd.AddValue("stage2Tpr", "stage-2 true-positive rate (measured branch recall)", stage2Tpr);
    cmd.AddValue("stage2Fpr", "stage-2 false-positive rate (measured branch FP/(FP+TN))", stage2Fpr);
    cmd.AddValue("stage2Mode", "independent | correlated", stage2Mode);
    cmd.AddValue("fogActiveW", "ASSUMED fog power while inspecting (W)", fogActiveW);
    cmd.AddValue("fogIdleW", "ASSUMED fog idle power (W)", fogIdleW);
    cmd.AddValue("flowmon", "install FlowMonitor", useFlowmon);
    cmd.Parse(argc, argv);

    RuleKind rule = ruleName == "ewma" ? RuleKind::EWMA : (ruleName == "criticality" ? RuleKind::CRITICALITY : RuleKind::FIXED);
    EwmaCoupling coupling = couplingName == "direct" ? EwmaCoupling::DIRECT : EwmaCoupling::INVERSE;
    mit.kind = mitigationName == "hysteresis" ? Mitigation::HYSTERESIS
             : mitigationName == "dual"       ? Mitigation::DUAL
             : mitigationName == "capacity"   ? Mitigation::CAPACITY
                                              : Mitigation::NONE;
    Stage2Config s2;
    s2.enabled = twoStage;
    s2.tpr = stage2Tpr;
    s2.fpr = stage2Fpr;
    s2.correlated = (stage2Mode == "correlated");

    RngSeedManager::SetSeed(42);
    RngSeedManager::SetRun(rngRun);
    std::mt19937 topoRng(1000003u * rngRun + 17u);

    NodeContainer wearables, attackers, fog, switches;
    wearables.Create(nWearables);
    attackers.Create(nAttackers);
    fog.Create(nFog);

    // --- switch fabric ---------------------------------------------------------------------
    uint32_t nSwitches = 1;
    std::vector<std::pair<uint32_t, uint32_t>> swLinks; // switch-switch links
    std::vector<double> swLinkDelayMs;
    if (topology == "clustered")
    {
        nSwitches = nFog + 1; // switch 0 is the core, 1..nFog are ward switches
        for (uint32_t k = 1; k <= nFog; ++k)
        {
            swLinks.push_back({0, k});
            swLinkDelayMs.push_back(1.0);
        }
    }
    else if (topology == "grid")
    {
        const uint32_t side = 4;
        nSwitches = side * side;
        for (uint32_t r = 0; r < side; ++r)
            for (uint32_t c = 0; c < side; ++c)
            {
                const uint32_t i = r * side + c;
                if (c + 1 < side) { swLinks.push_back({i, i + 1}); swLinkDelayMs.push_back(2.0); }
                if (r + 1 < side) { swLinks.push_back({i, i + side}); swLinkDelayMs.push_back(2.0); }
            }
    }
    else if (topology == "random")
    {
        // Random geometric graph on 16 switches in a unit square, radius 0.38, plus a nearest-
        // neighbour spanning chain to guarantee connectivity. Delay 1 ms + 9 ms per unit distance.
        nSwitches = 16;
        std::uniform_real_distribution<double> u(0.0, 1.0);
        std::vector<std::array<double, 2>> pos(nSwitches);
        for (auto& p : pos) p = {u(topoRng), u(topoRng)};
        auto dist = [&](uint32_t a, uint32_t b) { return std::hypot(pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]); };
        std::set<std::pair<uint32_t, uint32_t>> have;
        for (uint32_t a = 0; a < nSwitches; ++a)
            for (uint32_t b = a + 1; b < nSwitches; ++b)
                if (dist(a, b) < 0.38) { have.insert({a, b}); }
        // Prim-style connection of components
        std::vector<bool> in(nSwitches, false);
        in[0] = true;
        for (uint32_t step = 1; step < nSwitches; ++step)
        {
            double best = 1e9; uint32_t ba = 0, bb = 0;
            for (uint32_t a = 0; a < nSwitches; ++a) if (in[a])
                for (uint32_t b = 0; b < nSwitches; ++b) if (!in[b] && dist(a, b) < best) { best = dist(a, b); ba = a; bb = b; }
            in[bb] = true;
            have.insert({std::min(ba, bb), std::max(ba, bb)});
        }
        for (auto& e : have) { swLinks.push_back(e); swLinkDelayMs.push_back(1.0 + 9.0 * dist(e.first, e.second)); }
    }
    switches.Create(nSwitches);

    InternetStackHelper stack;
    stack.Install(wearables);
    stack.Install(attackers);
    stack.Install(fog);
    stack.Install(switches);

    PointToPointHelper access;
    access.SetDeviceAttribute("DataRate", StringValue("10Mbps"));
    access.SetChannelAttribute("Delay", StringValue("2ms"));
    PointToPointHelper backhaul;
    backhaul.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    backhaul.SetChannelAttribute("Delay", StringValue("1ms"));

    Ipv4AddressHelper ipv4;
    uint32_t subnet = 1;
    auto nextNet = [&]() {
        std::ostringstream net;
        net << "10." << (subnet / 254) << "." << (subnet % 254) << ".0";
        ipv4.SetBase(net.str().c_str(), "255.255.255.0");
        subnet++;
    };

    // Hop distances between switches (for nearest-fog assignment in grid/random).
    std::vector<std::vector<uint32_t>> adj(nSwitches);
    for (auto& e : swLinks) { adj[e.first].push_back(e.second); adj[e.second].push_back(e.first); }
    auto hops = [&](uint32_t from) {
        std::vector<uint32_t> d(nSwitches, 1u << 30);
        std::queue<uint32_t> q;
        d[from] = 0; q.push(from);
        while (!q.empty()) { uint32_t x = q.front(); q.pop(); for (auto y : adj[x]) if (d[y] > d[x] + 1) { d[y] = d[x] + 1; q.push(y); } }
        return d;
    };

    // Where each fog node and leaf attaches, and which fog node serves each leaf.
    std::vector<uint32_t> fogSwitch(nFog, 0);
    std::vector<uint32_t> leafSwitch(nWearables + nAttackers, 0);
    std::vector<uint32_t> leafFog(nWearables + nAttackers, 0);
    std::uniform_int_distribution<uint32_t> pickSw(0, nSwitches - 1);
    if (topology == "star")
    {
        for (uint32_t i = 0; i < nWearables; ++i) leafFog[i] = i % nFog;
        for (uint32_t i = 0; i < nAttackers; ++i) leafFog[nWearables + i] = (nWearables + i) % nFog;
    }
    else if (topology == "clustered")
    {
        // Wards: fog k sits on ward switch k+1. Wearables spread evenly over wards. Compromise is
        // localised: half of the attacker-bots are in ward 0, the rest spread over the others.
        for (uint32_t k = 0; k < nFog; ++k) fogSwitch[k] = k + 1;
        for (uint32_t i = 0; i < nWearables; ++i) { leafFog[i] = i % nFog; leafSwitch[i] = leafFog[i] + 1; }
        for (uint32_t i = 0; i < nAttackers; ++i)
        {
            uint32_t ward = (nFog == 1 || i < nAttackers / 2) ? 0 : 1 + (i % (nFog - 1));
            leafFog[nWearables + i] = ward;
            leafSwitch[nWearables + i] = ward + 1;
        }
    }
    else
    {
        // grid / random: fog nodes on distinct switches, leaves on random switches, each leaf
        // served by its nearest fog node in hops (ties broken by lower fog index).
        std::vector<uint32_t> order(nSwitches);
        for (uint32_t i = 0; i < nSwitches; ++i) order[i] = i;
        if (topology == "grid")
        {
            // spread fog nodes: corners first, then centre cells
            const uint32_t pref[16] = {0, 15, 3, 12, 5, 10, 6, 9, 1, 14, 2, 13, 4, 11, 7, 8};
            for (uint32_t k = 0; k < nFog; ++k) fogSwitch[k] = pref[k % 16];
        }
        else
        {
            std::shuffle(order.begin(), order.end(), topoRng);
            for (uint32_t k = 0; k < nFog; ++k) fogSwitch[k] = order[k % nSwitches];
        }
        std::vector<std::vector<uint32_t>> fogHops(nFog);
        for (uint32_t k = 0; k < nFog; ++k) fogHops[k] = hops(fogSwitch[k]);
        for (uint32_t i = 0; i < nWearables + nAttackers; ++i)
        {
            leafSwitch[i] = pickSw(topoRng);
            uint32_t best = 0;
            for (uint32_t k = 1; k < nFog; ++k) if (fogHops[k][leafSwitch[i]] < fogHops[best][leafSwitch[i]]) best = k;
            leafFog[i] = best;
        }
    }

    // fog <-> switch (created first, as in Review 2)
    std::vector<Ipv4Address> fogAddresses;
    for (uint32_t i = 0; i < nFog; ++i)
    {
        NetDeviceContainer d = backhaul.Install(NodeContainer(switches.Get(fogSwitch[i]), fog.Get(i)));
        nextNet();
        fogAddresses.push_back(ipv4.Assign(d).GetAddress(1));
    }
    // leaves <-> switch, wearables first
    std::vector<Ipv4Address> wearableAddresses, attackerAddresses;
    for (uint32_t i = 0; i < nWearables + nAttackers; ++i)
    {
        Ptr<Node> leaf = (i < nWearables) ? wearables.Get(i) : attackers.Get(i - nWearables);
        NetDeviceContainer d = access.Install(NodeContainer(switches.Get(leafSwitch[i]), leaf));
        nextNet();
        Ipv4Address a = ipv4.Assign(d).GetAddress(1);
        (i < nWearables ? wearableAddresses : attackerAddresses).push_back(a);
    }
    // switch <-> switch
    for (size_t e = 0; e < swLinks.size(); ++e)
    {
        PointToPointHelper sw;
        sw.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
        std::ostringstream dl;
        dl << swLinkDelayMs[e] << "ms";
        sw.SetChannelAttribute("Delay", StringValue(dl.str()));
        NetDeviceContainer d = sw.Install(NodeContainer(switches.Get(swLinks[e].first), switches.Get(swLinks[e].second)));
        nextNet();
        ipv4.Assign(d);
    }
    // controller address: switch 0 needs an address on some interface; use its first interface
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    const uint16_t port = 9000;
    std::vector<Ptr<FogIds>> idsApps;
    for (uint32_t i = 0; i < nFog; ++i)
    {
        Ptr<FogIds> app = CreateObject<FogIds>();
        app->Setup(port, MicroSeconds(idsLatencyUs), queueCapacity, rule, blockAfter, coupling, baseMs,
                   ewmaAlpha, mit, s2, 7919u * rngRun + 31u * i + 5u);
        app->m_phaseSplit = incrementalAt;
        fog.Get(i)->AddApplication(app);
        app->SetStartTime(Seconds(0.0));
        app->SetStopTime(Seconds(simTime));
        idsApps.push_back(app);
    }
    // Control plane: a controller sink on switch 0 plus every peer fog node.
    Ptr<CtrlSink> controller = CreateObject<CtrlSink>();
    switches.Get(0)->AddApplication(controller);
    controller->SetStartTime(Seconds(0.0));
    controller->SetStopTime(Seconds(simTime));
    std::vector<Ptr<CtrlSink>> fogSinks;
    for (uint32_t i = 0; i < nFog; ++i)
    {
        Ptr<CtrlSink> s = CreateObject<CtrlSink>();
        fog.Get(i)->AddApplication(s);
        s->SetStartTime(Seconds(0.0));
        s->SetStopTime(Seconds(simTime));
        fogSinks.push_back(s);
    }
    Ipv4Address controllerAddr = switches.Get(0)->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
    for (uint32_t i = 0; i < nFog; ++i)
    {
        std::vector<InetSocketAddress> peers;
        peers.push_back(InetSocketAddress(controllerAddr, 9100));
        for (uint32_t j = 0; j < nFog; ++j) if (j != i) peers.push_back(InetSocketAddress(fogAddresses[j], 9100));
        idsApps[i]->SetControlPeers(peers);
    }

    const auto weights = Weights(weightScheme);
    const auto classes = AssignClasses(nWearables, deviceMix);
    std::array<uint32_t, 4> classCount{};
    for (uint32_t i = 0; i < nWearables; ++i)
    {
        idsApps[leafFog[i]]->RegisterSource(wearableAddresses[i], true, weights[classes[i]]);
        classCount[classes[i]]++;
    }
    for (uint32_t i = 0; i < nAttackers; ++i)
    {
        idsApps[leafFog[nWearables + i]]->RegisterSource(attackerAddresses[i], false, 1.00);
    }

    std::vector<Ptr<JitteredSender>> wearableSenders;
    for (uint32_t i = 0; i < nWearables; ++i)
    {
        Ptr<JitteredSender> s = CreateObject<JitteredSender>();
        s->Setup(InetSocketAddress(fogAddresses[leafFog[i]], port), MilliSeconds(wearableIntervalMs), wearableJitter, 64,
                 Seconds(1.0), Seconds(simTime));
        wearables.Get(i)->AddApplication(s);
        s->SetStartTime(Seconds(1.0));
        s->SetStopTime(Seconds(simTime));
        wearableSenders.push_back(s);
    }
    const uint32_t nSlow = static_cast<uint32_t>(std::round(slowFraction * nAttackers));
    std::uniform_real_distribution<double> slowU(slowMinMs, slowMaxMs);
    std::vector<double> slowPeriod(nAttackers, -1.0);
    for (uint32_t i = 0; i < nSlow; ++i) slowPeriod[(i * nAttackers) / std::max(1u, nSlow)] = slowU(topoRng);
    for (uint32_t i = 0; i < nAttackers; ++i)
    {
        double startAt = (novelAttackAt > 0.0 && i % 2 == 1) ? novelAttackAt : 2.0;
        double intervalMs = (novelAttackAt > 0.0 && i % 2 == 1) ? 450.0 : attackerIntervalMs;
        double jit = 0.05;
        if (slowPeriod[i] > 0.0) { intervalMs = slowPeriod[i]; jit = slowJitter; }
        Ptr<JitteredSender> s = CreateObject<JitteredSender>();
        s->Setup(InetSocketAddress(fogAddresses[leafFog[nWearables + i]], port), MilliSeconds(intervalMs), jit, 64,
                 Seconds(startAt), Seconds(simTime));
        attackers.Get(i)->AddApplication(s);
        s->SetStartTime(Seconds(startAt));
        s->SetStopTime(Seconds(simTime));
    }
    if (incrementalAt > 0.0)
        for (auto& app : idsApps) Simulator::Schedule(Seconds(incrementalAt), &FogIds::ApplyIncrementalUpdate, app, incrementalBaseMs);

    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor;
    if (useFlowmon) monitor = flowmon.InstallAll();

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    // --- aggregate ---------------------------------------------------------------------------
    uint64_t received = 0, queueDrops = 0, blockedDrops = 0, inspected = 0, alerts = 0, tpA = 0, fpA = 0;
    uint64_t alertsBefore = 0, alertsAfter = 0, falseBefore = 0, falseAfter = 0, legitDelivered = 0, ctrlPk = 0, ctrlB = 0;
    uint32_t blockedAtk = 0, blockedWear = 0;
    double legitDelaySum = 0.0, maxLoad = 0.0, minThr = 1e9, ttbSum = 0.0;
    std::array<uint32_t, 4> blockedByClass{};
    std::vector<double> delays;
    for (auto& app : idsApps)
    {
        received += app->m_received; queueDrops += app->m_queueDrops; blockedDrops += app->m_blockedDrops;
        inspected += app->m_inspected; alerts += app->m_alerts; tpA += app->m_alertsOnAttackers; fpA += app->m_alertsOnWearables;
        alertsBefore += app->m_alertsBefore; alertsAfter += app->m_alertsAfter;
        falseBefore += app->m_falseAlertsBefore; falseAfter += app->m_falseAlertsAfter;
        legitDelivered += app->m_legitDelivered; legitDelaySum += app->m_legitDelaySum;
        delays.insert(delays.end(), app->m_legitDelays.begin(), app->m_legitDelays.end());
        ctrlPk += app->m_ctrlTxPackets; ctrlB += app->m_ctrlTxBytes;
        maxLoad = std::max(maxLoad, app->m_maxLoad); minThr = std::min(minThr, app->m_minThresholdMs);
        for (auto& e : app->m_sources)
            if (e.second.blocked)
            {
                if (e.second.legitimate) blockedWear++; else { blockedAtk++; ttbSum += e.second.blockedAt; }
            }
    }
    for (uint32_t i = 0; i < nWearables; ++i)
    {
        auto& st = idsApps[leafFog[i]]->m_sources[wearableAddresses[i]];
        if (st.blocked) blockedByClass[classes[i]]++;
    }
    uint64_t legitSent = 0;
    for (auto& s : wearableSenders) legitSent += s->m_sent;
    uint64_t ctrlRx = controller->m_rx;
    for (auto& s : fogSinks) ctrlRx += s->m_rx;
    std::sort(delays.begin(), delays.end());
    auto pct = [&](double q) { return delays.empty() ? 0.0 : delays[std::min(delays.size() - 1, (size_t)(q * delays.size()))]; };

    double netDelayMs = 0.0, netPdr = 0.0, thrMbps = 0.0;
    uint64_t txP = 0, rxP = 0, rxB = 0;
    if (useFlowmon)
    {
        monitor->CheckForLostPackets();
        double dsum = 0.0;
        for (auto& kv : monitor->GetFlowStats())
        {
            txP += kv.second.txPackets; rxP += kv.second.rxPackets; rxB += kv.second.rxBytes;
            dsum += kv.second.delaySum.GetSeconds();
        }
        netDelayMs = rxP ? 1000.0 * dsum / rxP : 0.0;
        netPdr = txP ? (double)rxP / txP : 0.0;
        thrMbps = 8.0 * rxB / simTime / 1e6;
    }
    const double busyS = inspected * idsLatencyUs * 1e-6;
    const double energyJ = nFog * fogIdleW * simTime + (fogActiveW - fogIdleW) * busyS;
    const double fnAttackers = nAttackers - blockedAtk;
    const bool latched = (nAttackers > 0) && (blockedAtk < 0.5 * nAttackers) && (queueDrops > 0);
    const uint64_t dataBytes = received * 64;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "RESULT,label=" << label << ",rule=" << ruleName << ",coupling=" << couplingName
              << ",mitigation=" << mitigationName << ",topology=" << topology << ",deviceMix=" << deviceMix
              << ",weightScheme=" << weightScheme << ",nWearables=" << nWearables << ",nAttackers=" << nAttackers
              << ",nFog=" << nFog << ",idsLatencyUs=" << idsLatencyUs << ",latencySource=" << latencySource
              << ",baseMs=" << baseMs << ",ewmaAlpha=" << ewmaAlpha << ",blockAfter=" << blockAfter
              << ",slowFraction=" << slowFraction << ",twoStage=" << twoStage << ",stage2Tpr=" << stage2Tpr
              << ",stage2Fpr=" << stage2Fpr << ",stage2Mode=" << stage2Mode << ",simTime=" << simTime
              << ",rngRun=" << rngRun
              << ",received=" << received << ",inspected=" << inspected << ",queueDrops=" << queueDrops
              << ",blockedDrops=" << blockedDrops << ",alerts=" << alerts << ",tpAlerts=" << tpA << ",fpAlerts=" << fpA
              << ",alertsBefore=" << alertsBefore << ",alertsAfter=" << alertsAfter << ",fpBefore=" << falseBefore
              << ",fpAfter=" << falseAfter
              << ",attackersBlocked=" << blockedAtk << ",wearablesBlocked=" << blockedWear << ",fnAttackers=" << fnAttackers
              << ",blockedLife=" << blockedByClass[0] << ",blockedHigh=" << blockedByClass[1]
              << ",blockedStd=" << blockedByClass[2] << ",blockedNon=" << blockedByClass[3]
              << ",nLife=" << classCount[0] << ",nHigh=" << classCount[1] << ",nStd=" << classCount[2] << ",nNon=" << classCount[3]
              << ",legitSent=" << legitSent << ",legitDelivered=" << legitDelivered
              << ",legitPdr=" << (legitSent ? (double)legitDelivered / legitSent : 0.0)
              << ",appDelayMeanMs=" << (legitDelivered ? 1000.0 * legitDelaySum / legitDelivered : 0.0)
              << ",appDelayP95Ms=" << 1000.0 * pct(0.95) << ",appDelayP99Ms=" << 1000.0 * pct(0.99)
              << ",netDelayMs=" << netDelayMs << ",netPdr=" << netPdr << ",throughputMbps=" << thrMbps
              << ",ctrlPackets=" << ctrlPk << ",ctrlBytes=" << ctrlB << ",ctrlRx=" << ctrlRx
              << ",ctrlOverheadPct=" << (dataBytes + ctrlB ? 100.0 * ctrlB / (dataBytes + ctrlB) : 0.0)
              << ",idsBusyS=" << busyS << ",energyJ=" << energyJ
              << ",meanTimeToBlockS=" << (blockedAtk ? ttbSum / blockedAtk - 2.0 : -1.0)
              << ",maxLoad=" << maxLoad << ",minThresholdMs=" << (minThr > 1e8 ? -1.0 : minThr)
              << ",latched=" << latched << "\n";
    Simulator::Destroy();
    return 0;
}
