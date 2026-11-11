"""Deterministic synthetic data for the payments lab.

One seed produces one coherent universe: accounts, merchants, transactions with fraud labels,
and the settlement inputs (payments, refunds, chargebacks, PSP file, injected breaks) derived from
those same transactions, so risk and settlement share ids.
"""
