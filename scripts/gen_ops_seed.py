"""Generate ``seed/migrations/003_seed_k8s_ops.sql`` — the operational schema plus
a fixed-seed synthetic dataset (~10k rows across 7 tables).

The SQL file is a committed build artefact so ``make migrate`` alone gives a
working database. It is *generated*, never hand-edited: change the shape here and
regenerate.

    uv run python scripts/gen_ops_seed.py            # rewrite the .sql file
    uv run python scripts/gen_ops_seed.py --check    # fail if the file is stale

Determinism: every random draw comes from one ``random.Random(SEED)`` consumed in
a fixed order, so the output is byte-for-byte stable across runs and machines.
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

SEED = 42

OUT_PATH = Path(__file__).resolve().parents[1] / "seed" / "migrations" / "003_seed_k8s_ops.sql"

# Row counts — tuned to ~10k total and to make the demo queries meaningful.
N_CLUSTERS = 15
N_NODES = 160
N_DEPLOYMENTS = 320
N_PODS = 2_000
N_INCIDENTS = 1_000
N_ALERTS = 5_000
N_ONCALL_LOGS = 1_500

# "now" for the synthetic world — fixed so generated timestamps never drift.
WORLD_NOW = datetime(2026, 9, 1, tzinfo=UTC)

_ROWS_PER_INSERT = 1_000

REGIONS = [
    "us-east-1",
    "us-east-2",
    "us-west-2",
    "eu-west-1",
    "eu-central-1",
    "ap-southeast-1",
    "ap-northeast-1",
    "sa-east-1",
]
ENVIRONMENTS = ["production", "staging", "development", "dr", "canary"]
K8S_VERSIONS = ["1.27.14", "1.28.10", "1.29.6", "1.30.2", "1.31.0"]

NODE_TYPES = ["control-plane", "worker", "worker", "worker", "worker", "gpu-worker"]
NODE_STATUSES = ["Ready"] * 22 + ["NotReady", "SchedulingDisabled"]
CPU_CHOICES = [4, 8, 16, 32, 64]

NAMESPACES = [
    "default",
    "kube-system",
    "monitoring",
    "ingress-nginx",
    "payments",
    "checkout",
    "search",
    "auth",
    "data-pipeline",
    "notifications",
]
APP_NAMES = [
    "api-gateway",
    "auth-service",
    "billing-worker",
    "cart-service",
    "checkout-api",
    "search-indexer",
    "recommendation-engine",
    "notification-dispatcher",
    "image-resizer",
    "payment-processor",
    "user-profile",
    "inventory-sync",
    "fraud-scorer",
    "email-relay",
    "webhook-fanout",
    "report-builder",
    "session-store",
    "feature-flags",
    "audit-logger",
    "metrics-aggregator",
]
POD_STATUSES = (
    ["Running"] * 78
    + ["Pending"] * 5
    + ["Succeeded"] * 5
    + ["CrashLoopBackOff"] * 4
    + ["ImagePullBackOff"] * 3
    + ["Terminating"] * 3
    + ["Failed"] * 2
)

SEVERITIES = ["P1"] * 6 + ["P2"] * 14 + ["P3"] * 30 + ["P4"] * 30 + ["P5"] * 20

RCA_TEMPLATES = [
    "Memory leak in {app} exhausted node memory; fixed by bumping limits and patching the leak.",
    "Rolling update of {app} shipped a bad config; rolled back to the previous revision.",
    "{app} overwhelmed the database connection pool; added a pgbouncer sidecar.",
    "Node {region} lost network connectivity; workloads rescheduled once the ENI recovered.",
    "Expired TLS certificate broke {app} ingress; renewed and automated rotation.",
    "Disk pressure on the {region} node group evicted {app} pods; expanded the volume.",
    "Upstream API rate-limited {app}; added backoff and a request budget.",
    "HPA thrashing on {app} caused repeated restarts; widened the stabilization window.",
    "DNS resolution failures inside kube-system degraded {app}; restarted CoreDNS.",
    "A noisy neighbour saturated CPU on the shared node; applied resource quotas.",
]

ALERT_NAMES = [
    ("KubePodCrashLooping", "availability"),
    ("KubeDeploymentReplicasMismatch", "availability"),
    ("KubePodNotReady", "availability"),
    ("TargetDown", "availability"),
    ("NodeMemoryPressure", "memory"),
    ("KubeContainerOOMKilled", "memory"),
    ("ContainerMemoryUsageHigh", "memory"),
    ("NodeCPUThrottlingHigh", "cpu"),
    ("KubeCPUOvercommit", "cpu"),
    ("NodeDiskPressure", "disk"),
    ("KubePersistentVolumeFillingUp", "disk"),
    ("NodeNetworkUnavailable", "network"),
    ("KubeletTooManyPods", "network"),
    ("HighRequestLatency", "network"),
    ("NetworkReceiveErrors", "network"),
]

ENGINEERS = [
    "adeyemi.k",
    "bianca.r",
    "chen.wei",
    "dmitri.s",
    "elena.p",
    "farhan.q",
    "grace.o",
    "hiroshi.t",
    "ingrid.m",
    "jamal.w",
    "kavya.n",
    "lucas.b",
]


def _q(value: str) -> str:
    """Single-quote a SQL string literal, escaping embedded quotes."""
    return "'" + value.replace("'", "''") + "'"


def _ts(dt: datetime) -> str:
    """Render a datetime as a UTC ``TIMESTAMPTZ`` literal."""
    return "'" + dt.strftime("%Y-%m-%d %H:%M:%S") + "+00'"


def _lit(value: object) -> str:
    """Render one Python value as its SQL literal (``bool`` before ``int``)."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, datetime):
        return _ts(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return _q(value)
    raise TypeError(f"unsupported literal: {value!r}")


def _insert_block(table: str, columns: list[str], rows: list[tuple[object, ...]]) -> list[str]:
    """Emit multi-row INSERTs for ``rows``, each idempotent via ``ON CONFLICT``.

    Rows carry explicit primary keys, so re-applying the migration onto an
    already-seeded database is a no-op rather than a duplicate-key error.
    """
    if not rows:
        return []
    out: list[str] = []
    collist = ", ".join(columns)
    for start in range(0, len(rows), _ROWS_PER_INSERT):
        chunk = rows[start : start + _ROWS_PER_INSERT]
        values = ",\n".join("  (" + ", ".join(_lit(v) for v in row) + ")" for row in chunk)
        out.append(f"INSERT INTO {table} ({collist}) VALUES\n{values}\nON CONFLICT DO NOTHING;")
    return out


_ID_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def _rand_id(rng: random.Random, length: int = 6) -> str:
    return "".join(rng.choice(_ID_ALPHABET) for _ in range(length))


def build_dataset(rng: random.Random) -> dict[str, list[tuple[object, ...]]]:
    """Build every table's rows, in dependency order, from one seeded RNG."""

    # --- clusters ---------------------------------------------------------
    clusters: list[tuple[object, ...]] = []
    cluster_ids: list[int] = []
    for i in range(1, N_CLUSTERS + 1):
        region = rng.choice(REGIONS)
        env = rng.choice(ENVIRONMENTS)
        name = f"k8s-{env}-{region}-{i:03d}"
        created = WORLD_NOW - timedelta(days=rng.randint(180, 900), seconds=rng.randint(0, 86_400))
        clusters.append((i, name, region, env, rng.choice(K8S_VERSIONS), created))
        cluster_ids.append(i)

    cluster_region = {row[0]: row[2] for row in clusters}

    # --- nodes -----------------------------------------------------------
    nodes: list[tuple[object, ...]] = []
    nodes_by_cluster: dict[int, list[int]] = {cid: [] for cid in cluster_ids}
    for node_id in range(1, N_NODES + 1):
        cid = rng.choice(cluster_ids)
        node_type = rng.choice(NODE_TYPES)
        cpu = rng.choice(CPU_CHOICES)
        memory = cpu * rng.choice([2, 4, 8])
        status = rng.choice(NODE_STATUSES)
        created = WORLD_NOW - timedelta(days=rng.randint(30, 700), seconds=rng.randint(0, 86_400))
        nodes.append((node_id, cid, node_type, cpu, memory, status, created))
        nodes_by_cluster[cid].append(node_id)

    # every cluster needs at least one node so pods can always be placed
    for cid, node_list in nodes_by_cluster.items():
        if not node_list:
            node_id = len(nodes) + 1
            nodes.append((node_id, cid, "worker", 8, 32, "Ready", WORLD_NOW - timedelta(days=200)))
            node_list.append(node_id)

    # --- deployments ---------------------------------------------------------
    deployments: list[tuple[object, ...]] = []
    # deployment_id -> (cluster_id, namespace, app_name)
    deploy_meta: dict[int, tuple[int, str, str]] = {}
    for dep_id in range(1, N_DEPLOYMENTS + 1):
        cid = rng.choice(cluster_ids)
        namespace = rng.choice(NAMESPACES)
        app = rng.choice(APP_NAMES)
        version = f"v{rng.randint(1, 6)}.{rng.randint(0, 12)}.{rng.randint(0, 20)}"
        image = f"registry.internal/{app}:{version}"
        replicas = rng.choice([1, 2, 2, 3, 3, 3, 4, 5, 6, 8, 10])
        created = WORLD_NOW - timedelta(days=rng.randint(1, 400), seconds=rng.randint(0, 86_400))
        deployments.append((dep_id, app, namespace, replicas, image, version, cid, created))
        deploy_meta[dep_id] = (cid, namespace, app)
    deployment_ids = list(deploy_meta)

    # --- pods -----------------------------------------------------------
    pods: list[tuple[object, ...]] = []
    pod_ids: list[int] = []
    for pod_id in range(1, N_PODS + 1):
        dep_id = rng.choice(deployment_ids)
        cid, namespace, app = deploy_meta[dep_id]
        name = f"{app}-{_rand_id(rng, 6)}-{_rand_id(rng, 5)}"
        status = rng.choice(POD_STATUSES)
        node_id = rng.choice(nodes_by_cluster[cid])
        if status == "CrashLoopBackOff":
            restarts = rng.randint(5, 240)
        elif status in ("Failed", "ImagePullBackOff"):
            restarts = rng.randint(1, 12)
        else:
            restarts = rng.choice([0, 0, 0, 0, 1, 2, 3])
        created = WORLD_NOW - timedelta(days=rng.randint(0, 60), seconds=rng.randint(0, 86_400))
        pods.append((pod_id, name, namespace, dep_id, status, node_id, restarts, created))
        pod_ids.append(pod_id)

    # --- incidents ----------------------------------------------------------
    incidents: list[tuple[object, ...]] = []
    incident_ids: list[int] = []
    incident_started: dict[int, datetime] = {}
    incident_resolved: dict[int, bool] = {}
    for inc_id in range(1, N_INCIDENTS + 1):
        cid = rng.choice(cluster_ids)
        started = WORLD_NOW - timedelta(days=rng.randint(0, 90), seconds=rng.randint(0, 86_400))
        severity = rng.choice(SEVERITIES)
        # higher severity → longer tail of resolution time
        ceiling = {"P1": 600, "P2": 400, "P3": 240, "P4": 180, "P5": 120}[severity]
        resolved_flag = rng.random() > 0.08  # ~8% still open
        mttr: int | None = None
        resolved_at: datetime | None = None
        rca: str | None = None
        if resolved_flag:
            mttr = rng.randint(5, ceiling)
            resolved_at = started + timedelta(minutes=mttr)
            rca = rng.choice(RCA_TEMPLATES).format(
                app=rng.choice(APP_NAMES), region=cluster_region[cid]
            )
        incidents.append((inc_id, severity, cid, started, resolved_at, mttr, rca))
        incident_ids.append(inc_id)
        incident_started[inc_id] = started
        incident_resolved[inc_id] = resolved_flag

    # --- alerts -----------------------------------------------------------
    alerts: list[tuple[object, ...]] = []
    for alert_id in range(1, N_ALERTS + 1):
        severity = rng.choice(SEVERITIES)
        source_pod = rng.choice(pod_ids)
        alertname, category = rng.choice(ALERT_NAMES)
        # ~35% of alerts are tied to a declared incident — those fire around its
        # start; the rest fire at any point in the 90-day window.
        alert_incident_id: int | None = None
        if rng.random() < 0.35:
            alert_incident_id = rng.choice(incident_ids)
            fired = incident_started[alert_incident_id] + timedelta(minutes=rng.randint(-15, 120))
            alert_resolved = incident_resolved[alert_incident_id]
        else:
            fired = WORLD_NOW - timedelta(days=rng.randint(0, 90), seconds=rng.randint(0, 86_400))
            alert_resolved = rng.random() > 0.25
        alerts.append(
            (
                alert_id,
                fired,
                severity,
                source_pod,
                alertname,
                category,
                alert_incident_id,
                alert_resolved,
            )
        )

    # --- oncall_logs -------------------------------------------------------
    oncall_logs: list[tuple[object, ...]] = []
    for log_id in range(1, N_ONCALL_LOGS + 1):
        inc_id = rng.choice(incident_ids)
        paged = incident_started[inc_id] + timedelta(seconds=rng.randint(0, 900))
        response_time = rng.randint(1, 45)
        escalated = rng.random() < 0.18
        oncall_logs.append((log_id, rng.choice(ENGINEERS), paged, inc_id, response_time, escalated))

    return {
        "clusters": clusters,
        "nodes": nodes,
        "deployments": deployments,
        "pods": pods,
        "incidents": incidents,
        "alerts": alerts,
        "oncall_logs": oncall_logs,
    }


_SCHEMA = """\
-- =============================================================================
-- 003_seed_k8s_ops.sql
--
-- The operational Kubernetes schema (7 tables) + a fixed-seed synthetic dataset
-- for the Text2SQL path and the advanced-RAG eval.
--
-- GENERATED FILE — do not edit by hand. Regenerate with:
--     uv run python scripts/gen_ops_seed.py
--
-- Idempotent like 001_create_users.sql: CREATE TABLE IF NOT EXISTS, and every
-- row is INSERT ... ON CONFLICT DO NOTHING against an explicit primary key, so
-- re-running `make migrate` is a no-op on an already-seeded database. To reseed
-- from scratch after changing the generator, DROP the seven tables first. The
-- users table is a separate migration and is never touched here.
-- =============================================================================

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id  INTEGER PRIMARY KEY,
    name        VARCHAR(128) UNIQUE NOT NULL,
    region      VARCHAR(32)  NOT NULL,
    environment VARCHAR(16)  NOT NULL,
    k8s_version VARCHAR(16)  NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS nodes (
    node_id    INTEGER PRIMARY KEY,
    cluster_id INTEGER NOT NULL REFERENCES clusters(cluster_id),
    node_type  VARCHAR(24) NOT NULL,
    cpu_cores  SMALLINT    NOT NULL,
    memory_gb  SMALLINT    NOT NULL,
    status     VARCHAR(24) NOT NULL DEFAULT 'Ready',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS deployments (
    deployment_id INTEGER PRIMARY KEY,
    name          VARCHAR(128) NOT NULL,
    namespace     VARCHAR(64)  NOT NULL,
    replicas      SMALLINT     NOT NULL DEFAULT 1,
    image         VARCHAR(256) NOT NULL,
    version       VARCHAR(32)  NOT NULL,
    cluster_id    INTEGER      NOT NULL REFERENCES clusters(cluster_id),
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pods (
    pod_id        INTEGER PRIMARY KEY,
    name          VARCHAR(160) NOT NULL,
    namespace     VARCHAR(64)  NOT NULL,
    deployment_id INTEGER      NOT NULL REFERENCES deployments(deployment_id),
    status        VARCHAR(32)  NOT NULL,
    node_id       INTEGER      NOT NULL REFERENCES nodes(node_id),
    restarts      INTEGER      NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS incidents (
    incident_id  INTEGER PRIMARY KEY,
    severity     VARCHAR(4)  NOT NULL,
    cluster_id   INTEGER     NOT NULL REFERENCES clusters(cluster_id),
    started_at   TIMESTAMPTZ NOT NULL,
    resolved_at  TIMESTAMPTZ,
    mttr_minutes INTEGER,
    rca_summary  TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id      INTEGER PRIMARY KEY,
    fired_at      TIMESTAMPTZ  NOT NULL,
    severity      VARCHAR(4)   NOT NULL,
    source_pod_id INTEGER      NOT NULL REFERENCES pods(pod_id),
    alertname     VARCHAR(128) NOT NULL,
    category      VARCHAR(16)  NOT NULL,
    incident_id   INTEGER      REFERENCES incidents(incident_id),
    resolved      BOOLEAN      NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS oncall_logs (
    log_id             INTEGER PRIMARY KEY,
    engineer           VARCHAR(64)  NOT NULL,
    paged_at           TIMESTAMPTZ  NOT NULL,
    incident_id        INTEGER      NOT NULL REFERENCES incidents(incident_id),
    response_time_mins INTEGER      NOT NULL,
    escalated          BOOLEAN      NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_nodes_cluster       ON nodes(cluster_id);
CREATE INDEX IF NOT EXISTS idx_deployments_cluster ON deployments(cluster_id);
CREATE INDEX IF NOT EXISTS idx_pods_deployment     ON pods(deployment_id);
CREATE INDEX IF NOT EXISTS idx_pods_node           ON pods(node_id);
CREATE INDEX IF NOT EXISTS idx_pods_status         ON pods(status);
CREATE INDEX IF NOT EXISTS idx_incidents_cluster   ON incidents(cluster_id);
CREATE INDEX IF NOT EXISTS idx_incidents_severity  ON incidents(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_fired_at     ON alerts(fired_at);
CREATE INDEX IF NOT EXISTS idx_alerts_severity     ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_category     ON alerts(category);
CREATE INDEX IF NOT EXISTS idx_alerts_incident     ON alerts(incident_id);
CREATE INDEX IF NOT EXISTS idx_oncall_incident     ON oncall_logs(incident_id);
"""

_COLUMNS: dict[str, list[str]] = {
    "clusters": ["cluster_id", "name", "region", "environment", "k8s_version", "created_at"],
    "nodes": [
        "node_id",
        "cluster_id",
        "node_type",
        "cpu_cores",
        "memory_gb",
        "status",
        "created_at",
    ],
    "deployments": [
        "deployment_id",
        "name",
        "namespace",
        "replicas",
        "image",
        "version",
        "cluster_id",
        "created_at",
    ],
    "pods": [
        "pod_id",
        "name",
        "namespace",
        "deployment_id",
        "status",
        "node_id",
        "restarts",
        "created_at",
    ],
    "incidents": [
        "incident_id",
        "severity",
        "cluster_id",
        "started_at",
        "resolved_at",
        "mttr_minutes",
        "rca_summary",
    ],
    "alerts": [
        "alert_id",
        "fired_at",
        "severity",
        "source_pod_id",
        "alertname",
        "category",
        "incident_id",
        "resolved",
    ],
    "oncall_logs": [
        "log_id",
        "engineer",
        "paged_at",
        "incident_id",
        "response_time_mins",
        "escalated",
    ],
}


def build_sql() -> str:
    rng = random.Random(SEED)
    dataset = build_dataset(rng)

    parts: list[str] = [_SCHEMA]
    for table in _COLUMNS:
        rows = dataset[table]
        parts.append(f"\n-- {table}: {len(rows):,} rows")
        parts.extend(_insert_block(table, _COLUMNS[table], rows))
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the committed .sql file differs from a fresh render",
    )
    args = parser.parse_args(argv)

    sql = build_sql()
    if args.check:
        current = OUT_PATH.read_text(encoding="utf-8") if OUT_PATH.exists() else ""
        if current != sql:
            print(f"{OUT_PATH.name} is stale — run: uv run python scripts/gen_ops_seed.py")
            return 1
        print(f"{OUT_PATH.name} is up to date")
        return 0

    OUT_PATH.write_text(sql, encoding="utf-8")
    print(f"wrote {OUT_PATH} ({len(sql):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
