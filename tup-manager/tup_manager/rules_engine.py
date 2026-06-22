import yaml
import re
import os
import glob
from typing import List, Optional, Any, Dict, Tuple
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import uuid

class RuleMatch(BaseModel):
    field: str
    patterns: List[str]

class FrameworkMapping(BaseModel):
    owasp_llm: str
    mitre_atlas: Optional[str] = None
    nist_ai_rmf: Optional[str] = None
    eu_ai_act: Optional[str] = None

class RuleResponse(BaseModel):
    action: str
    notify: bool
    alert_title: str

class RuleCorrelation(BaseModel):
    window: int = 5
    min_matches: int = 2
    distinct_patterns: Optional[int] = None

class RuleDefinition(BaseModel):
    id: str
    title: str
    description: str
    enabled: bool
    level: int = Field(ge=0, le=15)
    kind: str = "pattern"
    behavior: Optional[str] = None
    match: RuleMatch
    framework_mapping: FrameworkMapping
    response: RuleResponse
    correlation: Optional[RuleCorrelation] = None
    compiled_patterns: List[re.Pattern] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

BEHAVIOR_LABELS: Dict[str, str] = {
    "prompt_injection": "Prompt Injection",
    "authority_spoofing": "Authority Spoofing",
    "context_discovery": "Context Discovery",
    "safety_bypass": "Safety Bypass",
    "pii_exfiltration": "PII Exfiltration",
    "data_exposure": "Data Exposure",
    "toxic_output": "Toxic Output",
}

CAMPAIGN_SIGNATURES: List[Tuple[set, str]] = [
    ({"prompt_injection", "authority_spoofing", "context_discovery"}, "alignment_attack"),
    ({"prompt_injection", "context_discovery"}, "alignment_attack"),
    ({"authority_spoofing", "context_discovery"}, "alignment_attack"),
    ({"pii_exfiltration", "context_discovery"}, "data_exfiltration"),
    ({"data_exposure", "context_discovery"}, "credential_theft"),
    ({"safety_bypass", "toxic_output"}, "safety_degradation"),
    ({"safety_bypass", "prompt_injection"}, "jailbreak_campaign"),
]

class TUPRulesEngine:
    def __init__(self, rules_path: str):
        self.rules_path = rules_path
        self.rules: List[RuleDefinition] = []
        self.load_rules()

    def load_rules(self):
        self.rules = []
        pattern = os.path.join(self.rules_path, "**", "*.yml")
        for rule_file in sorted(glob.glob(pattern, recursive=True)):
            try:
                with open(rule_file, "r") as f:
                    data = yaml.safe_load(f)
                    rule = RuleDefinition(**data)
                    if rule.enabled and rule.match.patterns:
                        rule.compiled_patterns = [re.compile(p) for p in rule.match.patterns]
                    self.rules.append(rule)
            except Exception as e:
                print(f"Warning: Failed to load rule from {rule_file}: {e}")

    def _matches_any(self, content: str, rule: RuleDefinition) -> List[int]:
        matched_idx: List[int] = []
        for idx, pattern in enumerate(rule.compiled_patterns):
            if pattern.search(content):
                matched_idx.append(idx)
        return matched_idx

    def evaluate(self, event: Dict[str, Any], session_events: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        alerts = []
        for rule in self.rules:
            if not rule.enabled:
                continue
            field_to_check = rule.match.field
            content = event.get(field_to_check)

            if rule.kind == "correlation":
                if session_events is None or content is None:
                    continue
                corr = rule.correlation
                if corr is None:
                    continue
                window = max(1, corr.window)
                prior = list(session_events)
                window_events = (prior + [event])[-window:]
                matched_event_ids: List[str] = []
                matched_pattern_idxs: set = set()
                for ev in window_events:
                    ev_content = ev.get(field_to_check)
                    if ev_content is None:
                        continue
                    idxs = self._matches_any(ev_content, rule)
                    if idxs:
                        matched_event_ids.append(str(ev.get("event_id")))
                        matched_pattern_idxs.update(idxs)
                if len(matched_event_ids) < corr.min_matches:
                    continue
                if corr.distinct_patterns is not None and len(matched_pattern_idxs) < corr.distinct_patterns:
                    continue
                alert = {
                    "alert_id": str(uuid.uuid4()),
                    "event_id": str(event.get("event_id")),
                    "rule_id": rule.id,
                    "title": rule.response.alert_title,
                    "severity": rule.level,
                    "owasp_mapping": rule.framework_mapping.owasp_llm,
                    "mitre_atlas": rule.framework_mapping.mitre_atlas,
                    "matched_field": field_to_check,
                    "matched_value": f"correlation across {len(matched_event_ids)} events",
                    "action": rule.response.action,
                    "source": event.get("source"),
                    "model_id": event.get("model_id"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "correlation": True,
                    "correlated_events": matched_event_ids,
                    "distinct_patterns_matched": len(matched_pattern_idxs),
                    "behavior": rule.behavior,
                }
                alerts.append(alert)
                continue

            contents = self._field_variants(event, field_to_check)
            if not contents:
                continue

            matched = False
            for content in contents:
                for pattern in rule.compiled_patterns:
                    match = pattern.search(content)
                    if match:
                        alert = {
                            "alert_id": str(uuid.uuid4()),
                            "event_id": str(event.get("event_id")),
                            "rule_id": rule.id,
                            "title": rule.response.alert_title,
                            "severity": rule.level,
                            "owasp_mapping": rule.framework_mapping.owasp_llm,
                            "mitre_atlas": rule.framework_mapping.mitre_atlas,
                            "matched_field": field_to_check,
                            "matched_value": content[match.start():match.end()][:200],
                            "action": rule.response.action,
                            "source": event.get("source"),
                            "model_id": event.get("model_id"),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "behavior": rule.behavior,
                        }
                        alerts.append(alert)
                        matched = True
                        break
                if matched:
                    break
        return alerts

    def _field_variants(self, event: Dict[str, Any], field: str) -> List[str]:
        if field != "prompt":
            raw = event.get(field)
            return [raw] if raw else []

        seen: set = set()
        variants: List[str] = []
        for key in ("prompt", "prompt_normalized"):
            val = event.get(key)
            if val and val not in seen:
                seen.add(val)
                variants.append(val)
        for val in event.get("prompt_variants") or []:
            if val and val not in seen:
                seen.add(val)
                variants.append(val)
        return variants

    def assess_session(self, session_events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        ordered = sorted(session_events, key=lambda e: e.get("timestamp", ""))
        behavior_first_seen: Dict[str, str] = {}
        behavior_events: Dict[str, List[str]] = {}
        per_turn_behaviors: List[set] = []
        total_alerts = 0
        severity_sum = 0

        for ev in ordered:
            turn_behaviors: set = set()
            for a in ev.get("alerts", []):
                total_alerts += 1
                sev = a.get("severity", 0)
                severity_sum += sev
                b = a.get("behavior")
                if b:
                    turn_behaviors.add(b)
                    if b not in behavior_first_seen:
                        behavior_first_seen[b] = ev.get("timestamp", "")
                        behavior_events[b] = []
                    behavior_events[b].append(str(ev.get("event_id")))
            per_turn_behaviors.append(turn_behaviors)

        if not behavior_first_seen:
            return None

        attack_chain = sorted(behavior_first_seen.keys(), key=lambda b: behavior_first_seen[b])
        distinct_behaviors = len(attack_chain)
        turns_with_alerts = sum(1 for tb in per_turn_behaviors if len(tb) > 0)

        risk_score = min(100, severity_sum)
        risk_score += min(45, distinct_behaviors * 15)
        risk_score += min(40, max(0, turns_with_alerts - 1) * 10)
        multi_behavior_turns = sum(1 for tb in per_turn_behaviors if len(tb) > 1)
        risk_score += min(24, multi_behavior_turns * 8)
        risk_score = min(100, risk_score)

        if risk_score >= 90:
            severity_level = 15
            severity_label = "critical"
        elif risk_score >= 70:
            severity_level = 13
            severity_label = "high"
        elif risk_score >= 40:
            severity_level = 9
            severity_label = "medium"
        else:
            severity_level = 5
            severity_label = "low"

        behavior_set = set(attack_chain)
        campaign_type = "single_vector"
        if distinct_behaviors >= 2:
            campaign_type = "mixed_probe"
            for signature, label in CAMPAIGN_SIGNATURES:
                if signature.issubset(behavior_set):
                    campaign_type = label
                    break

        return {
            "attack_chain": attack_chain,
            "attack_chain_labels": [BEHAVIOR_LABELS.get(b, b) for b in attack_chain],
            "behavior_events": behavior_events,
            "risk_score": risk_score,
            "severity": severity_level,
            "severity_label": severity_label,
            "campaign_type": campaign_type,
            "distinct_behaviors": distinct_behaviors,
            "turns_with_alerts": turns_with_alerts,
            "total_alerts": total_alerts,
            "chain_length": len(ordered),
        }
