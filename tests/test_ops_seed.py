"""The operational schema + synthetic data migration (`003_seed_k8s_ops.sql`).

Two layers:
* the generator is deterministic and the committed `.sql` is not stale (pure);
* applied against a live Postgres, the schema and FK graph hold and the demo
  queries return real numbers (skips if Postgres is unreachable).
"""

from __future__ import annotations

from pathlib import Path

import psycopg2
import pytest

from app import db
from scripts import gen_ops_seed

_MIGRATIONS = Path(__file__).resolve().parents[1] / "seed" / "migrations"
_TABLES = ["clusters", "nodes", "deployments", "pods", "incidents", "alerts", "oncall_logs"]


# --- generator (pure) --------------------------------------------------------


def test_generator_is_deterministic() -> None:
    assert gen_ops_seed.build_sql() == gen_ops_seed.build_sql()


def test_committed_sql_is_not_stale() -> None:
    on_disk = gen_ops_seed.OUT_PATH.read_text(encoding="utf-8")
    assert on_disk == gen_ops_seed.build_sql(), (
        "003_seed_k8s_ops.sql is stale — run: uv run python scripts/gen_ops_seed.py"
    )


def test_sql_defines_all_seven_tables() -> None:
    sql = gen_ops_seed.OUT_PATH.read_text(encoding="utf-8")
    for table in _TABLES:
        assert f"CREATE TABLE {table} (" in sql


def test_sql_has_roughly_10k_rows() -> None:
    sql = gen_ops_seed.OUT_PATH.read_text(encoding="utf-8")
    row_lines = sum(1 for line in sql.splitlines() if line.startswith("  ("))
    assert 9_000 <= row_lines <= 11_000


# --- applied against Postgres ---------------------------------------------------


@pytest.fixture(scope="module")
def ops_db() -> None:
    try:
        for path in sorted(_MIGRATIONS.glob("*.sql")):
            db.run_sql_file(path)
    except psycopg2.OperationalError as exc:
        pytest.skip(f"Postgres not reachable: {exc}")


def _scalar(sql: str) -> object:
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()[0]


def test_every_table_is_populated(ops_db: None) -> None:
    for table in _TABLES:
        assert _scalar(f"SELECT count(*) FROM {table}") > 0


def test_incident_count_is_a_real_number(ops_db: None) -> None:
    assert _scalar("SELECT count(*) FROM incidents") == gen_ops_seed.N_INCIDENTS


def test_pods_sit_on_a_node_in_their_deployments_cluster(ops_db: None) -> None:
    orphans = _scalar(
        """
        SELECT count(*)
        FROM pods p
        JOIN nodes n ON p.node_id = n.node_id
        JOIN deployments d ON p.deployment_id = d.deployment_id
        WHERE n.cluster_id <> d.cluster_id
        """
    )
    assert orphans == 0


def test_resolved_incidents_have_an_mttr(ops_db: None) -> None:
    bad = _scalar(
        "SELECT count(*) FROM incidents WHERE resolved_at IS NOT NULL AND mttr_minutes IS NULL"
    )
    assert bad == 0


def test_demo_query_p1_incidents_by_cluster(ops_db: None) -> None:
    busiest = _scalar(
        """
        SELECT count(*) AS n
        FROM incidents
        WHERE severity = 'P1'
        GROUP BY cluster_id
        ORDER BY n DESC
        LIMIT 1
        """
    )
    assert busiest is not None and busiest > 0


def test_demo_query_avg_mttr_for_network_alerts(ops_db: None) -> None:
    avg = _scalar(
        """
        SELECT avg(i.mttr_minutes)
        FROM alerts a
        JOIN incidents i ON a.incident_id = i.incident_id
        WHERE a.category = 'network' AND i.mttr_minutes IS NOT NULL
        """
    )
    assert avg is not None and float(avg) > 0


def test_migration_is_rerunnable(ops_db: None) -> None:
    db.run_sql_file(_MIGRATIONS / "003_seed_k8s_ops.sql")
    assert _scalar("SELECT count(*) FROM incidents") == gen_ops_seed.N_INCIDENTS
