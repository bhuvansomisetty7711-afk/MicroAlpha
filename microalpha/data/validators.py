"""
microalpha/data/validators.py
==============================
Snapshot feed validation.

Validates a sequence of MarketSnapshots for:
  1. Strict monotonic timestamp ordering
  2. No crossed book (best_bid < best_ask)
  3. Positive sizes at all levels
  4. Bid/ask price ordering
  5. Timestamp positivity

A ValidationResult is returned per snapshot with severity codes:
  - ERROR:   snapshot should be dropped/rejected
  - WARNING: snapshot is suspicious but usable
  - OK:      snapshot passes all checks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from microalpha.data.snapshot import MarketSnapshot

log = logging.getLogger(__name__)


class Severity(Enum):
    OK      = "OK"
    WARNING = "WARNING"
    ERROR   = "ERROR"


@dataclass
class ValidationIssue:
    severity: Severity
    code:     str
    message:  str


@dataclass
class ValidationResult:
    snapshot:  MarketSnapshot
    issues:    List[ValidationIssue] = field(default_factory=list)

    @property
    def is_ok(self) -> bool:
        return all(i.severity != Severity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == Severity.WARNING for i in self.issues)

    @property
    def worst_severity(self) -> Severity:
        if any(i.severity == Severity.ERROR for i in self.issues):
            return Severity.ERROR
        if any(i.severity == Severity.WARNING for i in self.issues):
            return Severity.WARNING
        return Severity.OK


class SnapshotValidator:
    """
    Stateful validator that checks each incoming snapshot against invariants
    and tracks state across the snapshot stream (e.g., previous timestamp).

    Usage:
        validator = SnapshotValidator()
        valid_snaps = [r.snapshot for r in validator.validate_all(raw_snaps) if r.is_ok]
    """

    def __init__(self,
                 reject_crossed_book: bool = True,
                 min_bid_levels:      int   = 1,
                 min_ask_levels:      int   = 1,
                 max_timestamp_gap_s: float = 60.0):
        """
        reject_crossed_book: if True, crossed-book snapshots are ERROR (dropped).
        min_bid_levels:      minimum number of non-zero bid levels required.
        min_ask_levels:      minimum number of non-zero ask levels required.
        max_timestamp_gap_s: warn if gap between consecutive snapshots exceeds this.
        """
        self.reject_crossed_book  = reject_crossed_book
        self.min_bid_levels       = min_bid_levels
        self.min_ask_levels       = min_ask_levels
        self.max_timestamp_gap_s  = max_timestamp_gap_s
        self._prev_ts: Optional[float] = None
        self._counter: int = 0
        self._error_count: int = 0
        self._warning_count: int = 0

    def validate(self, snap: MarketSnapshot) -> ValidationResult:
        """Validate a single snapshot; update internal state."""
        result = ValidationResult(snapshot=snap)

        # ---- 1. Timestamp positivity ----------------------------------------
        if snap.ts < 0:
            result.issues.append(ValidationIssue(
                Severity.ERROR, "NEG_TIMESTAMP",
                f"Negative timestamp: {snap.ts}"
            ))

        # ---- 2. Monotonic timestamps ----------------------------------------
        if self._prev_ts is not None:
            if snap.ts < self._prev_ts:
                result.issues.append(ValidationIssue(
                    Severity.ERROR, "NON_MONOTONIC_TS",
                    f"ts {snap.ts} < previous ts {self._prev_ts} — clock went backwards"
                ))
            elif snap.ts == self._prev_ts:
                # Duplicate timestamp: WARNING (will get deterministic ordering by seq)
                result.issues.append(ValidationIssue(
                    Severity.WARNING, "DUPLICATE_TS",
                    f"Duplicate timestamp {snap.ts} — same-tick events will be ordered by sequence"
                ))
            # Large gap warning
            gap = snap.ts - self._prev_ts
            if gap > self.max_timestamp_gap_s:
                result.issues.append(ValidationIssue(
                    Severity.WARNING, "LARGE_TS_GAP",
                    f"Gap of {gap:.1f}s between snapshots — possible data dropout"
                ))

        # ---- 3. Minimum book depth ------------------------------------------
        usable_bids = [lv for lv in snap.bids if lv.size > 0]
        usable_asks = [lv for lv in snap.asks if lv.size > 0]
        if len(usable_bids) < self.min_bid_levels:
            result.issues.append(ValidationIssue(
                Severity.WARNING, "SPARSE_BIDS",
                f"Only {len(usable_bids)} non-zero bid levels (min {self.min_bid_levels})"
            ))
        if len(usable_asks) < self.min_ask_levels:
            result.issues.append(ValidationIssue(
                Severity.WARNING, "SPARSE_ASKS",
                f"Only {len(usable_asks)} non-zero ask levels (min {self.min_ask_levels})"
            ))

        # ---- 4. Crossed-book detection ---------------------------------------
        if snap.bids and snap.asks:
            best_bid = snap.bids[0].price
            best_ask = snap.asks[0].price
            if best_bid >= best_ask:
                sev = Severity.ERROR if self.reject_crossed_book else Severity.WARNING
                result.issues.append(ValidationIssue(
                    sev, "CROSSED_BOOK",
                    f"Crossed book: best_bid={best_bid} >= best_ask={best_ask}"
                ))

        # ---- 5. Trade aggressor values ---------------------------------------
        if snap.trade.aggressor not in (-1, 0, 1):
            result.issues.append(ValidationIssue(
                Severity.ERROR, "BAD_AGGRESSOR",
                f"trade.aggressor must be in {{-1,0,1}}, got {snap.trade.aggressor}"
            ))

        # ---- 6. Zero spread warning -----------------------------------------
        if snap.spread is not None and snap.spread == 0:
            result.issues.append(ValidationIssue(
                Severity.WARNING, "ZERO_SPREAD",
                f"Zero spread at {snap.ts}: best_bid == best_ask == {snap.best_bid}"
            ))

        # ---- Update state ----------------------------------------------------
        self._prev_ts = snap.ts
        self._counter += 1
        if result.worst_severity == Severity.ERROR:
            self._error_count += 1
        elif result.has_warnings:
            self._warning_count += 1

        return result

    def validate_all(self, snapshots: List[MarketSnapshot],
                     drop_errors: bool = True) -> List[ValidationResult]:
        """
        Validate entire snapshot feed, logging a summary at the end.

        If drop_errors=True, ERROR-flagged snapshots are excluded from the
        returned list (their ValidationResult objects are still returned for
        diagnostics via `results`).
        """
        results = [self.validate(s) for s in snapshots]
        log.info(
            "SnapshotValidator: %d snapshots, %d errors, %d warnings",
            self._counter, self._error_count, self._warning_count
        )
        if drop_errors:
            return [r for r in results if r.is_ok]
        return results

    @property
    def stats(self) -> dict:
        return {
            "total": self._counter,
            "errors": self._error_count,
            "warnings": self._warning_count,
        }
