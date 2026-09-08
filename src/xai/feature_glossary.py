"""Plain-language descriptions of IoTID20's flow features (Build-Instructions T3.3; TRD.md §4).

Why this exists
---------------
`TRD.md §4`'s output contract says the audience is "a hospital IT lead or clinician, not an ML
engineer", and T3.1's VERIFY block requires a reader be able to map every cited feature back to the
original data. T3.1 satisfied that by naming the original IoTID20 columns — `Bwd_Seg_Size_Avg`,
`Init_Bwd_Win_Byts`, `Fwd_IAT_Std` — and that was mistaken for readability.

**Traceable is not the same as comprehensible.** A column name lets an analyst find the field in the
dataset; it does not tell a clinician what was measured. `Init_Bwd_Win_Byts` is the TCP receive
window the device advertised on its first reply — nobody reaches that from the identifier. An
explanation naming only columns is legible to the person who built the model, which is precisely
the audience `PRD.md §4` says it is *not* for.

This module supplies a one-line description per feature, written for someone who understands
hospital networks but not machine learning or CICFlowMeter's naming scheme. `Explanation.to_summary`
uses them; `glossary_coverage` measures how much of a real explanation they actually cover, so the
gap between "traceable" and "readable" is a number rather than an assertion.

Conventions used in the descriptions
------------------------------------
    forward / "to the device"     : traffic from the initiator toward the destination device
    backward / "from the device"  : traffic returning from the device
    IAT                           : inter-arrival time, the gap between consecutive packets
    flow                          : one conversation between two endpoints
Direction is written as "to the device" / "from the device" rather than "forward"/"backward",
because the device is the thing the reader is protecting and the packet-capture convention is not
something they should have to learn.

Scope: the 62 features selected for IoTID20 (`docs/feature_selection_decision.md`). Edge-IIoTset
(T3.6) has a different schema and would need its own glossary; `glossary_coverage` reports any
feature it cannot describe rather than inventing one.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: Plain-language description per IoTID20 feature. Keys are the exact column names.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    # --- Endpoints and protocol ---------------------------------------------------------------
    "Src_Port": "the network port the traffic came from",
    "Dst_Port": "the network port on the device being contacted (identifies the service, e.g. web or telnet)",
    "Protocol": "which network protocol was used (TCP, UDP, or other)",

    # --- Flow timing --------------------------------------------------------------------------
    "Flow_Duration": "how long the conversation lasted",
    "Flow_IAT_Mean": "the average gap between packets in the conversation",
    "Flow_IAT_Std": "how irregular the gaps between packets were",
    "Flow_IAT_Max": "the longest pause between two packets",
    "Flow_IAT_Min": "the shortest gap between two packets",
    "Fwd_IAT_Tot": "total time spent waiting between packets sent to the device",
    "Fwd_IAT_Mean": "the average gap between packets sent to the device",
    "Fwd_IAT_Std": "how irregular the gaps were between packets sent to the device",
    "Fwd_IAT_Max": "the longest pause between packets sent to the device",
    "Fwd_IAT_Min": "the shortest gap between packets sent to the device",
    "Bwd_IAT_Tot": "total time spent waiting between replies from the device",
    "Bwd_IAT_Mean": "the average gap between replies from the device",
    "Bwd_IAT_Std": "how irregular the gaps were between replies from the device",
    "Bwd_IAT_Max": "the longest pause between replies from the device",
    "Bwd_IAT_Min": "the shortest gap between replies from the device",
    "Active_Max": "the longest continuous burst of activity before the conversation went quiet",
    "Active_Min": "the shortest burst of activity in the conversation",
    "Idle_Mean": "the average length of the quiet periods in the conversation",
    "Idle_Std": "how much the quiet periods varied in length",
    "Idle_Max": "the longest quiet period in the conversation",
    "Idle_Min": "the shortest quiet period in the conversation",

    # --- Volume and rate ----------------------------------------------------------------------
    "Flow_Byts/s": "how many bytes per second the conversation carried",
    "Flow_Pkts/s": "how many packets per second the conversation carried",
    "Fwd_Pkts/s": "how fast packets were being sent to the device",
    "Bwd_Pkts/s": "how fast the device was replying",
    "Tot_Fwd_Pkts": "how many packets were sent to the device",
    "Tot_Bwd_Pkts": "how many packets the device sent back",
    "TotLen_Fwd_Pkts": "the total amount of data sent to the device",
    "TotLen_Bwd_Pkts": "the total amount of data the device sent back",
    "Down/Up_Ratio": "how much the device received compared with how much it sent",
    "Fwd_Act_Data_Pkts": "how many packets sent to the device actually carried data rather than just control information",

    # --- Packet sizes -------------------------------------------------------------------------
    "Pkt_Len_Min": "the size of the smallest packet in the conversation",
    "Pkt_Len_Max": "the size of the largest packet in the conversation",
    "Pkt_Len_Mean": "the average packet size in the conversation",
    "Pkt_Len_Std": "how much packet sizes varied",
    "Pkt_Len_Var": "how much packet sizes varied (a second measure of the same thing)",
    "Pkt_Size_Avg": "the average size of packets in the conversation",
    "Fwd_Pkt_Len_Min": "the smallest packet sent to the device",
    "Fwd_Pkt_Len_Max": "the largest packet sent to the device",
    "Fwd_Pkt_Len_Mean": "the average size of packets sent to the device",
    "Fwd_Pkt_Len_Std": "how much the sizes of packets sent to the device varied",
    "Bwd_Pkt_Len_Min": "the smallest packet the device sent back",
    "Bwd_Pkt_Len_Max": "the largest packet the device sent back",
    "Bwd_Pkt_Len_Mean": "the average size of packets the device sent back",
    "Bwd_Pkt_Len_Std": "how much the sizes of the device's replies varied",
    "Fwd_Seg_Size_Avg": "the average amount of useful data per packet sent to the device",
    "Bwd_Seg_Size_Avg": "the average amount of useful data per packet the device sent back",
    "Fwd_Header_Len": "the total size of the addressing information on packets sent to the device",
    "Bwd_Header_Len": "the total size of the addressing information on the device's replies",
    "Subflow_Fwd_Pkts": "packets sent to the device in a typical exchange within the conversation",
    "Subflow_Bwd_Pkts": "packets sent back by the device in a typical exchange",
    "Subflow_Fwd_Byts": "data sent to the device in a typical exchange",
    "Subflow_Bwd_Byts": "data sent back by the device in a typical exchange",

    # --- Connection control flags -------------------------------------------------------------
    "SYN_Flag_Cnt": "how many packets were requests to open a new connection (a flood of these is how port scans and SYN attacks look)",
    "FIN_Flag_Cnt": "how many packets were requests to close a connection cleanly",
    "ACK_Flag_Cnt": "how many packets were acknowledgements of data already received",
    "PSH_Flag_Cnt": "how many packets asked for their data to be delivered immediately rather than buffered",
    "Bwd_PSH_Flags": "how many of the device's replies asked for immediate delivery",
    "Init_Bwd_Win_Byts": "how much data the device said it was ready to receive when it first replied",
}


def describe_feature(feature_name: str) -> str | None:
    """Return the plain-language description of a feature.

    Args:
        feature_name: the original column name.

    Returns:
        The description, or None if the feature has no entry. None is returned rather than a
        generated fallback: a made-up description of a network measurement would be worse than an
        honest gap, because the reader could not tell the difference.
    """
    return FEATURE_DESCRIPTIONS.get(feature_name)


def render_feature(feature_name: str) -> str:
    """Render a feature for an operator, with its description when one exists.

    Args:
        feature_name: the original column name.

    Returns:
        Either "description (Column_Name)" or the bare column name when undescribed. The column
        name is always kept so the explanation stays traceable to the data (T3.1's VERIFY block) —
        the description is added for readability, not substituted for provenance.
    """
    description = describe_feature(feature_name)
    return f"{description} ({feature_name})" if description else feature_name


def glossary_coverage(feature_names: list[str]) -> tuple[float, list[str]]:
    """Measure how much of a feature set the glossary can describe.

    Args:
        feature_names: the features to check.

    Returns:
        Tuple of the covered fraction and the sorted list of undescribed features.
    """
    if not feature_names:
        return 1.0, []
    missing = sorted({f for f in feature_names if f not in FEATURE_DESCRIPTIONS})
    covered = 1.0 - len(missing) / len(set(feature_names))
    if missing:
        logger.warning("%d features have no plain-language description: %s", len(missing), missing)
    return covered, missing
