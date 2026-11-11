-- Runs once, when the Postgres data volume is first created (docker-entrypoint-initdb.d).
-- One database per service: they must not share tables (ADR 0009).
--
-- An EXISTING volume skips this script. After pulling this change run, once:
--     docker compose down -v        # deletes the local Postgres volume
CREATE DATABASE risk_scoring;
CREATE DATABASE settlement;
