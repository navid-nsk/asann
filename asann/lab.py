"""Lab Diagnostic System for ASANN.

When basic diagnosis (vital signs) detects a disease but the root cause is
ambiguous, the scheduler can "refer to lab" — run targeted diagnostic tests
to differentiate between diseases that share the same symptoms.

Lab tests are hierarchical (Tier 1: cheap → Tier 3: expensive) and universal
(work for all task types and experiment tiers, not just PDE).

Additionally, PatientHistory provides persistent memory of past diagnoses,
treatments, and their outcomes — both within a training run (checkpoint) and
across runs (JSON file).
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import torch

from .diagnosis import DiseaseType, Diagnosis, HealthState
from .treatments import TreatmentType


# ==================== Patient History ====================


class TreatmentOutcome(Enum):
    """Outcome of a treatment after the recovery period."""
    IMPROVED = "improved"
    NO_CHANGE = "no_change"
    WORSENED = "worsened"
    HARMFUL = "harmful"   # Treatment made things dramatically worse (>2× loss)
    PENDING = "pending"


@dataclass
class DiagnosisRecord:
    """A single diagnosis event in the patient's history."""
    epoch: int
    health_state: str  # HealthState.name
    diseases: List[str]  # List of DiseaseType.name
    was_referred_to_lab: bool = False
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "health_state": self.health_state,
            "diseases": self.diseases,
            "was_referred_to_lab": self.was_referred_to_lab,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DiagnosisRecord:
        return cls(
            epoch=d["epoch"],
            health_state=d["health_state"],
            diseases=d["diseases"],
            was_referred_to_lab=d.get("was_referred_to_lab", False),
            timestamp=d.get("timestamp", 0.0),
        )


@dataclass
class TreatmentHistoryRecord:
    """A record of a treatment applied and its measured outcome."""
    epoch: int
    disease: str  # DiseaseType.name
    treatment: str  # TreatmentType.name
    pre_metric: float  # val_loss before treatment
    post_metric: Optional[float] = None  # val_loss after recovery
    outcome: str = "pending"  # TreatmentOutcome value
    escalation_level: int = 0

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "disease": self.disease,
            "treatment": self.treatment,
            "pre_metric": self.pre_metric,
            "post_metric": self.post_metric,
            "outcome": self.outcome,
            "escalation_level": self.escalation_level,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TreatmentHistoryRecord:
        return cls(**d)


@dataclass
class LabRecord:
    """A record of a lab referral and its results."""
    epoch: int
    report_summary: str  # Primary finding from lab report
    treatments_recommended: List[str]  # TreatmentType.name values
    treatment_chosen: Optional[str] = None  # What was actually applied
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "report_summary": self.report_summary,
            "treatments_recommended": self.treatments_recommended,
            "treatment_chosen": self.treatment_chosen,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LabRecord:
        return cls(**d)


@dataclass
class StabilityEvent:
    """Record of when training stability was achieved or broken."""
    epoch: int
    event: str  # "stabilized" or "broken"
    reason: str
    metric_value: float = 0.0

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "event": self.event,
            "reason": self.reason,
            "metric_value": self.metric_value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StabilityEvent:
        return cls(**d)


class PatientHistory:
    """Persistent memory of the model's medical history.

    Dual persistence:
    1. In checkpoint — survives resume within a training run
    2. Separate JSON file — survives across different training runs
       for the same task/dataset (like real medical records that follow
       the patient)

    Usage:
        history = PatientHistory()
        history.record_diagnosis(epoch, diagnosis)
        history.record_treatment(epoch, disease, treatment, pre_metric)
        history.evaluate_treatment(epoch, post_metric)

        # Query past experience
        history.was_treatment_effective(disease_type, treatment_type)
        history.should_refer_to_lab(diagnosis, config)
    """

    def __init__(self):
        # Diagnosis timeline
        self.diagnosis_log: List[DiagnosisRecord] = []
        # Treatment outcomes
        self.treatment_log: List[TreatmentHistoryRecord] = []
        # Lab results
        self.lab_log: List[LabRecord] = []
        # Architecture stability milestones
        self.stability_log: List[StabilityEvent] = []

        # Derived insights (updated after each treatment evaluation)
        self._effective_treatments: Dict[str, List[str]] = defaultdict(list)
        self._ineffective_treatments: Dict[str, List[str]] = defaultdict(list)

        # Disease recurrence counter (for lab referral decisions)
        self._disease_count: Dict[str, int] = defaultdict(int)

    # ---- Recording Methods ----

    def record_diagnosis(self, epoch: int, diagnosis: Diagnosis,
                         was_referred: bool = False):
        """Record a diagnosis event."""
        record = DiagnosisRecord(
            epoch=epoch,
            health_state=diagnosis.state.name,
            diseases=[d.disease_type.name for d in diagnosis.diseases],
            was_referred_to_lab=was_referred,
            timestamp=time.time(),
        )
        self.diagnosis_log.append(record)

        # Track disease recurrence
        for d in diagnosis.diseases:
            self._disease_count[d.disease_type.name] += 1

    def record_treatment(self, epoch: int, disease: DiseaseType,
                         treatment: TreatmentType, pre_metric: float,
                         escalation_level: int = 0):
        """Record that a treatment was applied. Call evaluate_treatment later."""
        record = TreatmentHistoryRecord(
            epoch=epoch,
            disease=disease.name,
            treatment=treatment.name,
            pre_metric=pre_metric,
            escalation_level=escalation_level,
        )
        self.treatment_log.append(record)

    def evaluate_treatment(self, post_metric: float,
                           improvement_threshold: float = 0.01,
                           current_epoch: int = 0,
                           min_delay: int = 0):
        """Evaluate the most recent pending treatment outcome.

        Args:
            post_metric: Current val metric (higher = better)
            improvement_threshold: Minimum relative improvement to count as "improved"
            current_epoch: Current epoch for delay check
            min_delay: Minimum epochs since treatment before evaluating.
                       Should be surgery_warmup_epochs so gate has fully ramped.
        """
        # Find the most recent pending treatment
        for record in reversed(self.treatment_log):
            if record.outcome == "pending":
                # Fix 2: Don't evaluate until gate ramp is complete
                if min_delay > 0 and current_epoch > 0:
                    if current_epoch - record.epoch < min_delay:
                        break  # Too early to evaluate — gate still ramping
                record.post_metric = post_metric
                pre = record.pre_metric

                if pre > 0:
                    relative_change = (pre - post_metric) / pre
                else:
                    relative_change = 0.0

                if relative_change > improvement_threshold:
                    record.outcome = TreatmentOutcome.IMPROVED.value
                    self._effective_treatments[record.disease].append(
                        record.treatment)
                elif relative_change < -improvement_threshold:
                    record.outcome = TreatmentOutcome.WORSENED.value
                    self._ineffective_treatments[record.disease].append(
                        record.treatment)
                else:
                    record.outcome = TreatmentOutcome.NO_CHANGE.value
                    self._ineffective_treatments[record.disease].append(
                        record.treatment)
                break

    def record_lab_result(self, epoch: int, report: LabReport,
                          treatment_chosen: Optional[TreatmentType] = None):
        """Record a lab referral and its results."""
        record = LabRecord(
            epoch=epoch,
            report_summary=report.primary_finding,
            treatments_recommended=[t.name for t in report.recommended_treatments],
            treatment_chosen=treatment_chosen.name if treatment_chosen else None,
            confidence=report.confidence,
        )
        self.lab_log.append(record)

    def record_stability_event(self, epoch: int, event: str, reason: str,
                               metric_value: float = 0.0):
        """Record a stability milestone (model stabilized or broken)."""
        self.stability_log.append(StabilityEvent(
            epoch=epoch, event=event, reason=reason,
            metric_value=metric_value,
        ))

    # ---- Query Methods ----

    def was_treatment_effective(self, disease: DiseaseType,
                                treatment: TreatmentType) -> Optional[bool]:
        """Check if a specific treatment worked for a disease before.

        Returns:
            True if improved, False if worsened/no_change, None if never tried
        """
        d_name = disease.name
        t_name = treatment.name
        if t_name in self._effective_treatments.get(d_name, []):
            return True
        if t_name in self._ineffective_treatments.get(d_name, []):
            return False
        return None

    def get_effective_treatments(self, disease: DiseaseType) -> List[str]:
        """Get list of treatments that worked for this disease type."""
        return list(set(self._effective_treatments.get(disease.name, [])))

    def get_ineffective_treatments(self, disease: DiseaseType) -> List[str]:
        """Get list of treatments that failed for this disease type.

        Returns ALL occurrences (not deduplicated) so callers can count
        how many times a treatment has failed.
        """
        return list(self._ineffective_treatments.get(disease.name, []))

    def disease_recurrence_count(self, disease: DiseaseType) -> int:
        """How many times this disease has been diagnosed."""
        return self._disease_count.get(disease.name, 0)

    def should_refer_to_lab(self, diagnosis: Diagnosis,
                            referral_recurrence: int = 2,
                            referral_on_ambiguous: bool = True) -> bool:
        """Determine if the current diagnosis warrants a lab referral.

        Criteria:
        1. Recurring disease: same disease diagnosed N+ times
        2. Ambiguous symptoms: multiple diseases detected simultaneously
        3. Treatment failure: past treatment for same disease made things worse
        4. High escalation: disease has reached escalation level 2+
        """
        if not diagnosis.is_sick:
            return False

        diseases = diagnosis.diseases

        # Criterion 1: Recurring disease
        for d in diseases:
            if self.disease_recurrence_count(d.disease_type) >= referral_recurrence:
                return True

        # Criterion 2: Ambiguous symptoms (multiple diseases)
        if referral_on_ambiguous and len(diseases) >= 2:
            return True

        # Criterion 3: Past treatment failure (worsened)
        for d in diseases:
            for record in reversed(self.treatment_log):
                if (record.disease == d.disease_type.name
                        and record.outcome == TreatmentOutcome.WORSENED.value):
                    return True

        # Criterion 4: High escalation (disease treated 2+ times already)
        for d in diseases:
            count = sum(1 for r in self.treatment_log
                        if r.disease == d.disease_type.name
                        and r.outcome != "pending")
            if count >= 2:
                return True

        return False

    def last_n_diagnoses(self, n: int = 5) -> List[DiagnosisRecord]:
        """Get the last N diagnosis records."""
        return self.diagnosis_log[-n:]

    def treatment_success_rate(self, disease: DiseaseType) -> float:
        """Compute success rate of treatments for a specific disease."""
        relevant = [r for r in self.treatment_log
                    if r.disease == disease.name and r.outcome != "pending"]
        if not relevant:
            return 0.0
        improved = sum(1 for r in relevant
                       if r.outcome == TreatmentOutcome.IMPROVED.value)
        return improved / len(relevant)

    # ---- Persistence: Checkpoint ----

    def state_dict(self) -> dict:
        """Serialize for checkpoint embedding."""
        return {
            "diagnosis_log": [r.to_dict() for r in self.diagnosis_log],
            "treatment_log": [r.to_dict() for r in self.treatment_log],
            "lab_log": [r.to_dict() for r in self.lab_log],
            "stability_log": [r.to_dict() for r in self.stability_log],
            "effective_treatments": dict(self._effective_treatments),
            "ineffective_treatments": dict(self._ineffective_treatments),
            "disease_count": dict(self._disease_count),
        }

    def load_state_dict(self, state: dict):
        """Restore from checkpoint."""
        self.diagnosis_log = [DiagnosisRecord.from_dict(d)
                              for d in state.get("diagnosis_log", [])]
        self.treatment_log = [TreatmentHistoryRecord.from_dict(d)
                              for d in state.get("treatment_log", [])]
        self.lab_log = [LabRecord.from_dict(d)
                        for d in state.get("lab_log", [])]
        self.stability_log = [StabilityEvent.from_dict(d)
                              for d in state.get("stability_log", [])]
        self._effective_treatments = defaultdict(
            list, state.get("effective_treatments", {}))
        self._ineffective_treatments = defaultdict(
            list, state.get("ineffective_treatments", {}))
        self._disease_count = defaultdict(
            int, state.get("disease_count", {}))

    # ---- Persistence: JSON file ----

    def save(self, filepath: str):
        """Export patient history to a JSON file for cross-run memory."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.state_dict(), f, indent=2)

    def load(self, filepath: str) -> bool:
        """Import patient history from a JSON file.

        Returns True if file was loaded, False if file not found.
        """
        if not os.path.exists(filepath):
            return False
        with open(filepath, "r") as f:
            state = json.load(f)
        self.load_state_dict(state)
        return True

    def __repr__(self) -> str:
        return (f"PatientHistory(diagnoses={len(self.diagnosis_log)}, "
                f"treatments={len(self.treatment_log)}, "
                f"lab_referrals={len(self.lab_log)})")


# ==================== Lab Test Framework ====================


@dataclass
class LabResult:
    """Result of a single lab test."""
    test_name: str
    findings: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    suggested_treatment: Optional[TreatmentType] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "findings": self.findings,
            "confidence": self.confidence,
            "suggested_treatment": (self.suggested_treatment.name
                                    if self.suggested_treatment else None),
            "evidence": {k: v for k, v in self.evidence.items()
                         if not isinstance(v, torch.Tensor)},
        }


@dataclass
class LabReport:
    """Collection of lab test results with aggregated conclusions."""
    results: List[LabResult] = field(default_factory=list)
    primary_finding: str = ""
    recommended_treatments: List[TreatmentType] = field(default_factory=list)
    confidence: float = 0.0

    def add_result(self, result: LabResult):
        """Add a test result and update recommendations."""
        self.results.append(result)
        self._update_recommendations()

    def _update_recommendations(self):
        """Aggregate recommendations from all test results."""
        if not self.results:
            return

        # Rank treatments by confidence across all tests
        treatment_confidence: Dict[TreatmentType, float] = {}
        for result in self.results:
            if result.suggested_treatment is not None:
                t = result.suggested_treatment
                # Keep highest confidence for each treatment
                if t not in treatment_confidence or result.confidence > treatment_confidence[t]:
                    treatment_confidence[t] = result.confidence

        # Sort by confidence (highest first)
        sorted_treatments = sorted(treatment_confidence.items(),
                                   key=lambda x: x[1], reverse=True)
        self.recommended_treatments = [t for t, _ in sorted_treatments]

        # Primary finding = highest confidence result
        best = max(self.results, key=lambda r: r.confidence)
        self.primary_finding = (f"{best.test_name}: "
                                f"{best.findings.get('summary', 'see details')}")
        self.confidence = best.confidence

    def to_dict(self) -> dict:
        return {
            "results": [r.to_dict() for r in self.results],
            "primary_finding": self.primary_finding,
            "recommended_treatments": [t.name for t in self.recommended_treatments],
            "confidence": self.confidence,
        }


class LabTest(ABC):
    """Base class for all lab diagnostic tests.

    Each test has:
    - name: Human-readable test name
    - cost_level: 1 (cheap), 2 (medium), 3 (expensive)
    - applicable_diseases: Which diseases this test helps diagnose
    """

    name: str = "BaseLabTest"
    cost_level: int = 1  # 1=cheap, 2=medium, 3=expensive

    @abstractmethod
    def applicable_diseases(self) -> List[DiseaseType]:
        """Which disease types this test is relevant for."""
        ...

    def is_applicable(self, diagnosis: Diagnosis,
                      task_type: str = "regression",
                      patient_history: Optional[PatientHistory] = None) -> bool:
        """Check if this test should run for the given diagnosis."""
        if not diagnosis.is_sick:
            return False
        diagnosed_diseases = {d.disease_type for d in diagnosis.diseases}
        return bool(diagnosed_diseases & set(self.applicable_diseases()))

    @abstractmethod
    def run(self, model: torch.nn.Module,
            val_data: Any,
            signals: Dict[str, Any],
            config: Any,
            task_type: str = "regression",
            patient_history: Optional['PatientHistory'] = None,
            **kwargs) -> LabResult:
        """Execute the lab test and return results.

        Args:
            model: The ASANN model to test
            val_data: Validation DataLoader (or batch)
            signals: Current diagnostic signals (GDS stats, etc.)
            config: ASANNConfig
            task_type: "regression" or "classification"
            patient_history: Past treatment outcomes for history-aware decisions
        """
        ...


class LabDiagnostics:
    """Orchestrates lab test execution with hierarchical escalation.

    Runs tests from cheapest (Tier 1) to most expensive (Tier 3),
    stopping early when confidence is high enough.

    Usage:
        lab = LabDiagnostics()
        lab.register_test(ResidualStructureTest())
        lab.register_test(GradientConcentrationTest())
        ...

        if patient_history.should_refer_to_lab(diagnosis):
            report = lab.run_tests(model, diagnosis, val_data, signals, config)
    """

    def __init__(self, max_tier: int = 2, confidence_threshold: float = 0.7):
        self._tests: List[LabTest] = []
        self.max_tier = max_tier
        self.confidence_threshold = confidence_threshold

    def register_test(self, test: LabTest):
        """Register a lab test."""
        self._tests.append(test)

    @property
    def registered_tests(self) -> List[LabTest]:
        return list(self._tests)

    def run_tests(self, model: torch.nn.Module,
                  diagnosis: Diagnosis,
                  val_data: Any,
                  signals: Dict[str, Any],
                  config: Any,
                  task_type: str = "regression",
                  patient_history: Optional[PatientHistory] = None,
                  ) -> LabReport:
        """Run applicable lab tests in tier order.

        Starts with cheapest tests (Tier 1) and escalates to more expensive
        tests if confidence is below threshold.

        Args:
            model: The ASANN model
            diagnosis: Current diagnosis
            val_data: Validation data
            signals: Diagnostic signals (GDS, etc.)
            config: ASANNConfig
            task_type: "regression" or "classification"
            patient_history: Past medical records for context
        """
        report = LabReport()

        # Sort tests by cost level (cheapest first)
        applicable_tests = [
            t for t in self._tests
            if (t.cost_level <= self.max_tier
                and t.is_applicable(diagnosis, task_type, patient_history))
        ]
        applicable_tests.sort(key=lambda t: t.cost_level)

        current_tier = 0
        for test in applicable_tests:
            # If we've moved to a higher tier and confidence is already high, stop
            if (test.cost_level > current_tier
                    and current_tier > 0
                    and report.confidence >= self.confidence_threshold):
                break

            current_tier = test.cost_level
            try:
                result = test.run(model, val_data, signals, config,
                                  task_type=task_type,
                                  patient_history=patient_history)
                report.add_result(result)
            except Exception:
                # Lab test failure should not crash training
                pass

        return report
