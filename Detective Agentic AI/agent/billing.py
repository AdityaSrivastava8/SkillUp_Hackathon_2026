import os
import json
import re
import time
import math
from typing import Tuple, List, Dict, Any

# File Paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
PAYMENTS_FILE = os.path.join(DATA_DIR, "payments.json")

# Status Constants
STATUS_PENDING = "PENDING"
STATUS_APPROVED = "APPROVED"
STATUS_PARTIAL = "PARTIAL"
STATUS_TOPUP_DONE = "TOPUP_DONE"
STATUS_FLAGGED = "FLAGGED"
STATUS_INVALID_UTR = "INVALID_UTR"


def _ensure_data_dir():
    """Ensures that the data directory exists."""
    os.makedirs(DATA_DIR, exist_ok=True)


def load_payments() -> List[Dict[str, Any]]:
    """Loads payment records from JSON file."""
    _ensure_data_dir()
    if not os.path.exists(PAYMENTS_FILE):
        return []
    try:
        with open(PAYMENTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_payments(payments: List[Dict[str, Any]]):
    """Saves payment records to JSON file."""
    _ensure_data_dir()
    with open(PAYMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(payments, f, indent=2, ensure_ascii=False)


def validate_utr_format(utr: str) -> bool:
    """
    Validates Indian standard UPI UTR / RRN numbers.
    Standard UTR numbers are strictly 12 digits.
    """
    cleaned_utr = utr.strip()
    return bool(re.match(r"^\d{12}$", cleaned_utr))


def submit_payment(
    plan: str,
    required_amount: float,
    amount_paid: float,
    evals: Any,
    utr: str
) -> Tuple[str, str, float]:
    """
    Submits and processes a payment proof:
    1. Validates UTR structure (returns error if fake/invalid format).
    2. Checks for duplicate UTR submissions.
    3. Calculates remaining deficit balance.
    4. Queues a full payment for admin verification.
    """
    utr_clean = utr.strip()

    # Step 1: Validate UTR format
    if not validate_utr_format(utr_clean):
        return (
            STATUS_INVALID_UTR,
            "❌ Invalid UTR ID! Bank transaction reference numbers must be exactly 12 numerical digits.",
            required_amount
        )

    if not isinstance(amount_paid, (int, float)) or not math.isfinite(amount_paid) or amount_paid <= 0:
        return STATUS_FLAGGED, "❌ Invalid payment amount.", required_amount

    payments = load_payments()

    # Step 2: Check for duplicate UTR submissions
    for pmt in payments:
        if (
            pmt.get("utr") == utr_clean
            or utr_clean in pmt.get("topup_utrs", [])
        ):
            return STATUS_FLAGGED, "⚠️ This UTR ID has already been submitted.", 0.0

    # Step 3: Calculate balance
    remaining = max(0.0, required_amount - amount_paid)

    if amount_paid >= required_amount:
        status = STATUS_PENDING
        msg = f"✅ Payment proof submitted for admin verification. The claimed ₹{amount_paid:,.0f} is not confirmed until approval."
    else:
        status = STATUS_PARTIAL
        msg = f"⚠️ Partial payment proof submitted for review. You claimed ₹{amount_paid:,.0f} of ₹{required_amount:,.0f}; the remaining ₹{remaining:,.0f} is based on your claim and is not verified."

    new_record = {
        "utr": utr_clean,
        "plan": plan,
        "required_amount": required_amount,
        "amount_paid": amount_paid,
        "remaining_balance": remaining,
        "evals": evals,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "topup_utrs": []
    }

    payments.append(new_record)
    save_payments(payments)
    return status, msg, remaining


def submit_topup(topup_utr: str, parent_utr: str, topup_amount: float) -> Tuple[str, str, float]:
    """
    Records a top-up claim for admin verification.
    """
    topup_utr_clean = topup_utr.strip()

    if not validate_utr_format(topup_utr_clean):
        return (
            STATUS_INVALID_UTR,
            "❌ Invalid Top-Up UTR ID! Must be a valid 12-digit bank reference number.",
            0.0
        )

    if not isinstance(topup_amount, (int, float)) or not math.isfinite(topup_amount) or topup_amount <= 0:
        return STATUS_FLAGGED, "❌ Invalid top-up amount.", 0.0

    payments = load_payments()

    # A UTR must be unique across original payments and all top-ups.
    if any(
        pmt.get("utr") == topup_utr_clean
        or topup_utr_clean in pmt.get("topup_utrs", [])
        for pmt in payments
    ):
        return STATUS_FLAGGED, "⚠️ This UTR ID has already been submitted.", 0.0

    for pmt in payments:
        if pmt.get("utr") == parent_utr:
            if pmt.get("status") not in [STATUS_PARTIAL, STATUS_TOPUP_DONE]:
                return STATUS_FLAGGED, "⚠️ This payment is not awaiting a top-up.", pmt.get("remaining_balance", 0.0)
            if pmt.get("remaining_balance", 0.0) <= 0:
                return STATUS_FLAGGED, "⚠️ Full payment proof is already awaiting admin verification.", 0.0
            if "topup_utrs" not in pmt:
                pmt["topup_utrs"] = []

            # Update totals
            pmt["topup_utrs"].append(topup_utr_clean)
            pmt["amount_paid"] += topup_amount
            new_remaining = max(0.0, pmt["required_amount"] - pmt["amount_paid"])
            pmt["remaining_balance"] = new_remaining

            if new_remaining == 0.0:
                pmt["status"] = STATUS_TOPUP_DONE
                msg = f"✅ Top-up proof submitted for admin verification. The claimed ₹{topup_amount:,.0f} is not confirmed until approval."
            else:
                pmt["status"] = STATUS_PARTIAL
                msg = f"⚠️ Top-up proof submitted for review. The claimed remaining balance is ₹{new_remaining:,.0f}."

            save_payments(payments)
            return pmt["status"], msg, new_remaining

    return STATUS_FLAGGED, "❌ Original payment record not found.", 0.0


def approve_payment(utr: str) -> bool:
    """Manually approves a payment record from the admin panel."""
    payments = load_payments()
    for pmt in payments:
        if pmt.get("utr") == utr:
            pmt["status"] = STATUS_APPROVED
            pmt["remaining_balance"] = 0.0
            save_payments(payments)
            return True
    return False


def flag_partial(utr: str) -> bool:
    """Flags a payment record as partial/incomplete."""
    payments = load_payments()
    for pmt in payments:
        if pmt.get("utr") == utr:
            pmt["status"] = STATUS_PARTIAL
            save_payments(payments)
            return True
    return False


def get_pending_payments() -> List[Dict[str, Any]]:
    """Returns all payment submissions awaiting review or completion."""
    payments = load_payments()
    return [p for p in payments if p.get("status") in [STATUS_PENDING, STATUS_PARTIAL, STATUS_TOPUP_DONE]]


def get_partial_by_utr(utr: str) -> Dict[str, Any]:
    """Retrieves a specific partial payment record by UTR."""
    payments = load_payments()
    for p in payments:
        if p.get("utr") == utr and p.get("status") in [STATUS_PARTIAL, STATUS_TOPUP_DONE]:
            return p
    return {} 
