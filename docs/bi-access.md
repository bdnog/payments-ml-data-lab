# Querying the marts directly (DuckDB)

> **All data is synthetic.**

The marts are plain Hive-partitioned Parquet under `<marts_dir>/<mart>/settlement_date=YYYY-MM-DD/`.
Any tool that reads Parquet can query them without the API: DuckDB, Polars, Spark, Trino, or a BI tool
with a Parquet or DuckDB connector. DuckDB is the quickest way to try it.

```bash
make settle-all DATA=data/samples            # writes data/marts
uv run --with duckdb python                  # or: pip install duckdb, or the duckdb CLI
```

Two things to know:

- Pass `hive_partitioning=true` and the glob `**/*.parquet`. DuckDB then exposes `settlement_date` as
  a column from the path and **skips the partitions a filter on it excludes**.
- **Money is in minor units** (centavos): divide by 100 for reais, in SQL if you like.

`<MARTS>` below is your marts directory (the tests substitute a real one and run every query on this
page, so they cannot rot).

## What do we owe each merchant on a day?

`daily_settlement_by_merchant` is the **ledger-expected payout**, not what the PSP confirmed.

```sql
SELECT merchant_id, payment_count, gross_minor / 100.0 AS gross, net_minor / 100.0 AS net
FROM read_parquet('<MARTS>/daily_settlement_by_merchant/**/*.parquet', hive_partitioning=true)
WHERE settlement_date = DATE '2026-01-15'
ORDER BY net_minor DESC
LIMIT 10;
```

## Total expected payout per week

```sql
SELECT date_trunc('week', settlement_date) AS week,
       sum(net_minor) / 100.0 AS expected_payout,
       sum(refunds_minor + chargebacks_minor) / 100.0 AS clawbacks
FROM read_parquet('<MARTS>/daily_settlement_by_merchant/**/*.parquet', hive_partitioning=true)
GROUP BY 1
ORDER BY 1;
```

## Which merchants have the most negative days?

A negative net means refunds and chargebacks exceeded that day's sales.

```sql
SELECT merchant_id, count(*) AS negative_days, sum(net_minor) / 100.0 AS total_negative_net
FROM read_parquet('<MARTS>/daily_settlement_by_merchant/**/*.parquet', hive_partitioning=true)
WHERE net_minor < 0
GROUP BY merchant_id
ORDER BY total_negative_net
LIMIT 10;
```

## How many reconciliation breaks, and of what kind?

```sql
SELECT break_type, count(*) AS breaks, sum(diff_minor) / 100.0 AS net_difference
FROM read_parquet('<MARTS>/reconciliation_breaks/**/*.parquet', hive_partitioning=true)
GROUP BY break_type
ORDER BY breaks DESC;
```

`diff_minor` is **PSP minus ledger**: negative means the PSP paid less than the ledger expected.

## The largest open breaks of a run date

```sql
SELECT break_id, break_type, payment_id, ledger_amount_minor, psp_amount_minor, diff_minor
FROM read_parquet('<MARTS>/reconciliation_breaks/**/*.parquet', hive_partitioning=true)
WHERE run_date = DATE '2026-01-15'
ORDER BY abs(diff_minor) DESC
LIMIT 10;
```

## Are we being charged the contracted fee?

`fee_summary` only has sales that match the PSP on both sides with the *same gross*, so a non-zero
`fee_diff_minor` is purely a fee error.

```sql
SELECT merchant_id,
       sum(expected_fee_minor) / 100.0 AS expected_fee,
       sum(actual_fee_minor) / 100.0 AS actual_fee,
       sum(fee_diff_minor) / 100.0 AS overcharged
FROM read_parquet('<MARTS>/fee_summary/**/*.parquet', hive_partitioning=true)
GROUP BY merchant_id
HAVING sum(fee_diff_minor) <> 0
ORDER BY overcharged DESC;
```

## Did a run happen on a date?

A date that was run and found nothing has an **empty** partition; a date that was never run has none.
The partition directories are the answer:

```sql
SELECT DISTINCT settlement_date
FROM read_parquet('<MARTS>/reconciliation_breaks/**/*.parquet', hive_partitioning=true)
ORDER BY 1;
```

Dates with zero breaks appear in the directory listing but not in this result, because an empty file
has no rows. List the folders (`ls <MARTS>/reconciliation_breaks`) or ask the API: `GET
/v1/exports/reconciliation_breaks/{date}` is `200` (possibly empty) when the run happened and `404`
when it never did.

## Through the API instead

`GET /v1/exports/{mart}/{date}` returns one partition as a Parquet file, and DuckDB reads it directly:

```python
import duckdb

duckdb.sql("SELECT * FROM read_parquet('fee_summary-2026-01-15.parquet') LIMIT 5").show()
```
