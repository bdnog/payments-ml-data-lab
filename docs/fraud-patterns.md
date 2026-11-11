# Fraud patterns, hard negatives and labels

How fraud is simulated, what makes it hard to detect, and why the label a model can learn from is not
the ground truth. The last section lists the points worth making in an interview.

## Fraud episodes

Fraud comes in *episodes*: a burst of related transactions with one purpose. Every transaction of an
episode is labeled `is_fraud` and carries its `fraud_pattern`. The mix below is a share of the fraud
*transactions* (a profile's fraud rate times its target volume), and every pattern appears at least
once even in small profiles.

| Pattern | Share | Shape |
| --- | --- | --- |
| `CARD_TESTING` | 30% | 5 to 30 tiny charges (0.50 to 5.00 BRL) within 15 minutes from one device, mostly web, at many different merchants (so the card is mostly new for each). With probability 0.3 it ends with one larger charge. |
| `ACCOUNT_TAKEOVER` | 30% | Account older than 90 days. A **new device**, usually a foreign IP and likely a proxy. Amount 3x to 10x the account's average, in high-risk categories (electronics, jewelry, games, direct marketing, money transfer). 1 to 5 transactions within 2 hours. |
| `GEO_MISMATCH` | 25% | A Brazilian card used from a foreign IP through a proxy, on an account under 30 days old. Moderate amounts (1x to 3x the category's typical ticket). 1 to 4 transactions within 3 hours. |
| `VELOCITY_BURST` | 15% | 8 to 20 ordinary-looking purchases within an hour across many merchants, a quarter of them in person. |

The attacker brings a **fresh device** to each episode, so `device_is_new_for_account` fires on the
first transaction only. Later ones in the same episode are not new, which is how it works with real
data too.

## Hard negatives

Legitimate behavior that looks like fraud, so the problem is not trivially separable:

- **Travelers**: a few accounts spend 3 to 10 days abroad, so their IP country is foreign, without a proxy.
- **Big-ticket purchases**: about 0.6% of purchases cost 4x to 8x what the account usually spends, on a known device.
- **Legitimate bursts**: 3 to 6 purchases within 30 minutes at one merchant, on a known device,
  with ordinary amounts. A burst looks like a velocity burst by counts alone.
- **Brand-new accounts with normal behavior**: 15% of accounts open during the window, so account age is
  not a fraud signal on its own.
- **Ordinary noise**: about 2% of legitimate online users are behind a VPN, and about 1% of their IPs
  geolocate to another country.

## Overlap, on purpose

Fraud is not always loud. About 15% of account-takeover and geo-mismatch episodes are *stealthy*:
a domestic IP and no proxy. Together with the hard negatives, this means no single feature separates
the classes, and a model has to combine signals. A test asserts that stealthy fraud and noisy
legitimate traffic both exist.

## Labels: delay, bias and what is unknown

`is_fraud` is the ground truth, available because the data is synthetic. **It is for analysis only,
never a feature, and not what a model should be trained on**: in production nobody knows it. What
exists in production is a *report*:

1. **Declines are hidden.** About 6% of legitimate and 40% of fraudulent transactions are declined
   upstream and never become payments. The decision is not a column, because it would leak the label.
2. **Only approved fraud can be charged back.** Declined fraud costs nothing and is never reported.
   Voided payments (about 3%) move no money either.
3. **Most, not all, approved fraud is reported.** About 85% of it gets
   `chargeback_reported_at`; the rest is never reported. Those transactions are fraud labeled as
   legitimate.
4. **Reports arrive late.** The delay is 7 to 45 days after the transaction, so many reports land
   *after the end of the data window*. That is intended.

A training label is therefore derived from a **cutoff date**: a transaction is positive if
`chargeback_reported_at` is on or before the cutoff. A test shows the count of positives grows with the
cutoff.

```python
label = (pl.col("chargeback_reported_at") <= cutoff).fill_null(False)
```

## Talking points

- **Label delay.** Recent data looks cleaner than it is, because the chargebacks have not arrived. Train on a
  cutoff that leaves the last 45 days out, or the model learns that recent transactions are legitimate.
  Split train and test by time, never at random: a random split lets the model see the future.
- **Selection bias.** The model is trained on transactions that *were approved*, because only those can
  produce a label. The part of the world the previous system already declined is missing, so the model
  can never learn from it. Evaluating on approved traffic only flatters whoever decided before.
- **Unreported fraud.** About 15% of approved fraud is a negative in the training data. Labels are noisy,
  and the noise is correlated with the very patterns that are hardest to see.
- **Metrics.** At a 1% fraud rate, accuracy is meaningless (predicting "legitimate" scores 99%).
  Use precision-recall (PR-AUC) and recall at a fixed false-positive rate, and look at the cost of each kind of
  error, not only the count.
- **Point-in-time features.** A feature computed with data from after the transaction makes the model look
  great offline and fail online. Here the velocity features exclude the current transaction and the future,
  and tests prove it ([ADR 0006](adr/0006-v1-scoring-takes-precomputed-behavior-features.md)).
- **Ground truth versus label.** Having `is_fraud` lets you measure how wrong the practical label is (what
  fraction of fraud is reported by a given cutoff) and what that costs a model, something real data never allows.
