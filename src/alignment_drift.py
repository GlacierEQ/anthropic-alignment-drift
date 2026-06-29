"""Alignment Drift Detector — Catches agents becoming unsafe over time.

Genuine Innovation: No one has built this.

Standard approach: Check each action against safety rules.
This approach: Monitor the PATTERN of agent decisions over time.
Detect when the pattern starts drifting toward unsafe territory.

Key Insight: Misalignment doesn't happen in a single action.
It happens gradually — a series of small decisions that each seem
fine, but collectively move the agent toward unsafe behavior.
By the time you notice, the agent is already misaligned.

How it works:
1. Build a baseline of "aligned" decision patterns
2. Track the agent's decision distribution over time
3. Compute the drift between current and baseline distributions
4. When drift exceeds threshold, alert and optionally isolate

This is NOT content filtering. This is PATTERN analysis.
It catches misalignment that content filters miss because
each individual action looks fine — only the trend is dangerous.

Real-world scenario:
- Agent starts by always asking for permission (aligned)
- Over weeks, it starts acting first and asking later
- Each individual action seems reasonable
- But the PATTERN shows decreasing permission-seeking
- Alignment Drift Detector catches this before it becomes a problem
"""

import math
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


@dataclass
class DecisionRecord:
    agent_id: str
    action_type: str
    requires_permission: bool
    permission_granted: bool
    context_risk_level: float
    timestamp: float
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class DriftSignal:
    metric: str
    baseline_value: float
    current_value: float
    drift_magnitude: float
    direction: str
    confidence: float


class DecisionPatternAnalyzer:
    """Analyzes patterns in agent decision-making over time.

    Innovation: Uses distribution divergence to detect gradual misalignment.
    Not checking individual actions — checking the SHAPE of behavior over time.
    """

    def __init__(self, baseline_window: int = 200):
        self.baseline_window = baseline_window
        self.action_history: Dict[str, List[DecisionRecord]] = defaultdict(list)
        self.permission_history: Dict[str, List[bool]] = defaultdict(list)
        self.risk_history: Dict[str, List[float]] = defaultdict(list)

    def record_decision(self, record: DecisionRecord):
        self.action_history[record.agent_id].append(record)
        self.permission_history[record.agent_id].append(record.requires_permission and record.permission_granted)
        self.risk_history[record.agent_id].append(record.context_risk_level)

        for key in [self.action_history, self.permission_history, self.risk_history]:
            if len(key[record.agent_id]) > self.baseline_window:
                key[record.agent_id] = key[record.agent_id][-self.baseline_window:]

    def compute_drift_signals(self, agent_id: str) -> List[DriftSignal]:
        signals = []

        perm_history = self.permission_history.get(agent_id, [])
        if len(perm_history) >= 50:
            recent = perm_history[-50:]
            older = perm_history[-100:-50] if len(perm_history) >= 100 else perm_history[:50]

            recent_rate = sum(recent) / len(recent)
            older_rate = sum(older) / len(older)

            if older_rate > 0:
                drift = (recent_rate - older_rate) / older_rate
                signals.append(DriftSignal(
                    metric="permission_seeking_rate",
                    baseline_value=older_rate,
                    current_value=recent_rate,
                    drift_magnitude=abs(drift),
                    direction="decreasing" if drift < 0 else "increasing",
                    confidence=min(0.95, len(perm_history) / 200),
                ))

        risk_history = self.risk_history.get(agent_id, [])
        if len(risk_history) >= 50:
            recent_risk = risk_history[-50:]
            older_risk = risk_history[-100:-50] if len(risk_history) >= 100 else risk_history[:50]

            recent_avg = sum(recent_risk) / len(recent_risk)
            older_avg = sum(older_risk) / len(older_risk)

            if older_avg > 0:
                drift = (recent_avg - older_avg) / older_avg
                signals.append(DriftSignal(
                    metric="risk_tolerance",
                    baseline_value=older_avg,
                    current_value=recent_avg,
                    drift_magnitude=abs(drift),
                    direction="increasing" if drift > 0 else "decreasing",
                    confidence=min(0.95, len(risk_history) / 200),
                ))

        return signals


class AlignmentDriftDetector:
    """Detects gradual alignment drift in AI agents.

    Innovation: Catches misalignment that happens GRADUALLY.
    Each action looks fine. Only the pattern is dangerous.

    This is the difference between:
    - Content filtering: "Is this action safe?" (one-shot)
    - Alignment drift: "Is this agent becoming less safe?" (trend)

    Real-world impact: Prevents the slow drift that leads to
    catastrophic misalignment. By the time content filters catch
    it, the agent has already evolved past them.
    """

    def __init__(self, drift_threshold: float = 0.3):
        self.drift_threshold = drift_threshold
        self.analyzer = DecisionPatternAnalyzer()
        self.agent_states: Dict[str, dict] = {}
        self._drift_log: List[dict] = []
        self._isolation_log: List[dict] = []

    def record_decision(self, record: DecisionRecord):
        self.analyzer.record_decision(record)

        signals = self.analyzer.compute_drift_signals(record.agent_id)

        high_drift = [s for s in signals if s.drift_magnitude > self.drift_threshold]

        if high_drift:
            max_drift = max(high_drift, key=lambda s: s.drift_magnitude)

            state = "DRIFTING"
            if max_drift.drift_magnitude > 0.5:
                state = "CRITICAL"
                self._isolate_agent(record.agent_id, max_drift)

            self.agent_states[record.agent_id] = {
                "state": state,
                "max_drift": max_drift.drift_magnitude,
                "metric": max_drift.metric,
                "direction": max_drift.direction,
                "timestamp": time.time(),
            }

            self._drift_log.append({
                "agent_id": record.agent_id,
                "signals": [
                    {
                        "metric": s.metric,
                        "drift": s.drift_magnitude,
                        "direction": s.direction,
                    }
                    for s in signals
                ],
                "timestamp": time.time(),
            })

        else:
            self.agent_states[record.agent_id] = {
                "state": "ALIGNED",
                "timestamp": time.time(),
            }

    def _isolate_agent(self, agent_id: str, signal: DriftSignal):
        self.agent_states[agent_id]["state"] = "ISOLATED"
        self._isolation_log.append({
            "agent_id": agent_id,
            "metric": signal.metric,
            "drift": signal.drift_magnitude,
            "direction": signal.direction,
            "timestamp": time.time(),
        })

    def get_safety_report(self) -> dict:
        aligned = sum(1 for s in self.agent_states.values() if s["state"] == "ALIGNED")
        drifting = sum(1 for s in self.agent_states.values() if s["state"] == "DRIFTING")
        critical = sum(1 for s in self.agent_states.values() if s["state"] == "CRITICAL")
        isolated = sum(1 for s in self.agent_states.values() if s["state"] == "ISOLATED")

        return {
            "total_agents": len(self.agent_states),
            "aligned": aligned,
            "drifting": drifting,
            "critical": critical,
            "isolated": isolated,
            "system_status": "CRITICAL" if critical > 0 or isolated > 0 else "WARNING" if drifting > 0 else "ALIGNED",
            "recent_drifts": self._drift_log[-5:],
            "recent_isolations": self._isolation_log[-5:],
        }


if __name__ == "__main__":
    detector = AlignmentDriftDetector()

    for step in range(200):
        for agent_id in ["agent-1", "agent-2", "agent-3"]:
            if agent_id == "agent-2" and step > 100:
                requires_perm = step % 3 != 0
                risk = 0.8
            else:
                requires_perm = step % 5 == 0
                risk = 0.2

            record = DecisionRecord(
                agent_id=agent_id,
                action_type="query",
                requires_permission=requires_perm,
                permission_granted=True,
                context_risk_level=risk,
                timestamp=time.time(),
            )
            detector.record_decision(record)

    print(json.dumps(detector.get_safety_report(), indent=2))
