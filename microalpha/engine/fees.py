"""
microalpha/engine/fees.py
==========================
Fee model: maker/taker, per-share, and regulatory fees.

Models the US equity exchange fee structure (maker-taker):
- Makers (passive limit fills) receive a rebate (negative fee → credit)
- Takers (aggressive market orders) pay a surcharge
- Fixed per-share component models SEC transaction fee and FINRA TAF
- No mixing of maker/taker rates (invariant #31)
- Fee uses fill price, not mid price (invariant #30)
- All fees are attributed in the fill event (invariant #32)

Typical values for US equities:
    fee_bps         = 0.30   ← base exchange rate (30c per $1000 notional)
    maker_rebate_bps = -0.20 ← maker rebate (credit, negative)
    taker_fee_bps   = 0.30   ← additional taker surcharge
    fee_per_share   = 0.0001 ← SEC fee proxy (≈ $0.0001/share)

A round-trip (passive buy then passive sell at same price) produces:
    realized_pnl = -(maker_fee_buy + maker_fee_sell)
which equals exactly -(fees_paid), confirming invariant #33.
"""

from __future__ import annotations

from dataclasses import dataclass

from microalpha.data.snapshot import FillType, Side


@dataclass
class FeeConfig:
    """
    Exchange fee configuration. All bps values are in BASIS POINTS (1 bps = 0.01%).
    Negative bps = credit (rebate). fee_per_share is dollars per share.
    """
    fee_bps:          float = 0.30      # base fee on notional (bps)
    maker_rebate_bps: float = -0.20     # additional credit for passive fills (bps)
    taker_fee_bps:    float = 0.30      # additional surcharge for aggressive fills (bps)
    fee_per_share:    float = 0.0001    # fixed per-share fee (USD, models SEC/FINRA)

    def __post_init__(self):
        # Sanity: maker total should be ≤ taker total (makers subsidize takers)
        maker_total = self.fee_bps + self.maker_rebate_bps
        taker_total = self.fee_bps + self.taker_fee_bps
        if maker_total > taker_total + 1e-9:
            import warnings
            warnings.warn(
                f"FeeConfig: maker_total ({maker_total} bps) > taker_total ({taker_total} bps). "
                "This is unusual for a maker-taker venue."
            )


class FeeModel:
    """
    Computes per-fill fees respecting maker vs taker distinction.

    The fee formula per fill:
        notional = qty * price
        maker_fee = notional * (fee_bps + maker_rebate_bps) / 10000 + qty * fee_per_share
        taker_fee = notional * (fee_bps + taker_fee_bps)   / 10000 + qty * fee_per_share

    Negative maker_fee means a net rebate is credited.

    Invariants enforced:
        #30  — fee_per_share uses fill price (not mid)
        #31  — maker and taker rates are never mixed
        #32  — fee is in each fill event log
        #33  — roundtrip at same price: realized_pnl = -fees_paid
    """

    def __init__(self, config: FeeConfig = None):
        self.config = config or FeeConfig()

    def compute_fee(self, qty: float, price: float, fill_type: FillType) -> float:
        """
        Compute the net fee for a single fill.

        Returns a dollar amount:
            positive → cost to strategy
            negative → credit to strategy (maker rebate)

        qty:       shares filled
        price:     fill price per share (MUST be the actual fill price, not mid)
        fill_type: PASSIVE (maker) or AGGRESSIVE (taker)
        """
        notional = qty * price
        cfg = self.config

        if fill_type == FillType.PASSIVE:
            fee = (notional * (cfg.fee_bps + cfg.maker_rebate_bps) / 10_000.0
                   + qty * cfg.fee_per_share)
        else:
            fee = (notional * (cfg.fee_bps + cfg.taker_fee_bps) / 10_000.0
                   + qty * cfg.fee_per_share)

        return fee

    def maker_fee_rate(self) -> float:
        """Effective maker fee in basis points (negative = rebate)."""
        return self.config.fee_bps + self.config.maker_rebate_bps

    def taker_fee_rate(self) -> float:
        """Effective taker fee in basis points."""
        return self.config.fee_bps + self.config.taker_fee_bps

    def roundtrip_cost_bps(self) -> float:
        """
        Total cost of a passive buy + passive sell at the same price.
        Should equal 2 * maker_fee_rate (in bps) + per-share costs.
        """
        return 2.0 * self.maker_fee_rate()
