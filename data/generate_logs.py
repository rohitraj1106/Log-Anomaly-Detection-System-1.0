"""
Synthetic Log Data Generator.

Generates realistic distributed system logs for testing and demonstration.
Simulates logs from multiple services with configurable anomaly injection.
"""

import random
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Services and their typical log patterns
SERVICES = [
    "auth-service",
    "api-gateway",
    "payment-service",
    "user-service",
    "notification-service",
    "search-service",
]

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LEVEL_WEIGHTS_NORMAL = [0.10, 0.60, 0.15, 0.12, 0.03]
LEVEL_WEIGHTS_ANOMALOUS = [0.02, 0.15, 0.20, 0.45, 0.18]

NORMAL_MESSAGES = {
    "auth-service": [
        "User login successful for user_id={uid}",
        "Token refreshed for session={sid}",
        "Password validation passed for user_id={uid}",
        "OAuth callback received from provider=google",
        "Session created with TTL=3600s",
        "MFA verification successful for user_id={uid}",
        "API key validated for client={client}",
    ],
    "api-gateway": [
        "Request routed to {service} — latency={lat}ms",
        "Rate limit check passed for client={client}",
        "Circuit breaker status=CLOSED for {service}",
        "Health check passed for upstream={service}",
        "TLS handshake completed in {lat}ms",
        "Request {method} {path} completed — status={status}",
    ],
    "payment-service": [
        "Payment processed: amount={amount} currency=USD txn_id={txn}",
        "Refund initiated: txn_id={txn} amount={amount}",
        "Payment method verified for user_id={uid}",
        "Invoice generated: invoice_id={inv}",
        "Subscription renewed for user_id={uid}",
    ],
    "user-service": [
        "User profile updated: user_id={uid}",
        "User preferences saved for user_id={uid}",
        "Email verification sent to user_id={uid}",
        "User search query processed in {lat}ms",
        "Avatar uploaded for user_id={uid} size={size}KB",
    ],
    "notification-service": [
        "Email sent to user_id={uid} template={template}",
        "Push notification delivered to device={device}",
        "SMS sent to user_id={uid} provider=twilio",
        "Notification batch processed: count={count}",
        "Webhook delivered to endpoint={endpoint}",
    ],
    "search-service": [
        "Search query processed: query='{query}' results={count} latency={lat}ms",
        "Index updated: documents={count} duration={lat}ms",
        "Cache hit for query hash={hash}",
        "Search suggestion generated for prefix='{prefix}'",
        "Relevance model scored {count} documents in {lat}ms",
    ],
}

ANOMALOUS_MESSAGES = {
    "auth-service": [
        "ALERT: Brute force attack detected from IP={ip} — {count} failed attempts in 60s",
        "CRITICAL: Authentication service unresponsive — timeout after 30000ms",
        "SECURITY: Suspicious login from unusual geography IP={ip} user_id={uid}",
        "ERROR: Token validation failed — invalid signature detected",
        "ALERT: Mass password reset requests detected — {count} requests in 5 minutes",
        "CRITICAL: Credential stuffing attack detected from botnet — {count} unique IPs",
        "ERROR: SSO provider unreachable — cascading authentication failures",
    ],
    "api-gateway": [
        "CRITICAL: Upstream {service} returned 503 — circuit breaker OPEN",
        "ERROR: Request timeout exceeded 30s for {service} — request dropped",
        "ALERT: DDoS pattern detected — {count} requests/sec from IP={ip}",
        "ERROR: TLS certificate validation failed for upstream={service}",
        "CRITICAL: Memory usage at 95% — initiating emergency GC",
        "ERROR: Connection pool exhausted for {service} — {count} pending requests",
    ],
    "payment-service": [
        "CRITICAL: Payment gateway timeout — txn_id={txn} amount={amount} STUCK",
        "ALERT: Fraudulent transaction pattern detected — user_id={uid} amount={amount}",
        "ERROR: Double charge detected — txn_id={txn} refund required",
        "CRITICAL: Payment reconciliation mismatch — delta={amount}",
        "ERROR: PCI compliance check failed — encryption key expired",
    ],
    "user-service": [
        "ERROR: Database connection pool exhausted — {count} queries queued",
        "CRITICAL: User data corruption detected — user_id={uid} checksum mismatch",
        "ALERT: Mass account deletion request — {count} accounts flagged",
        "ERROR: Profile service latency spike — p99={lat}ms (10x normal)",
        "CRITICAL: Replication lag exceeded threshold — lag={lat}ms",
    ],
    "notification-service": [
        "CRITICAL: Email provider rate limited — {count} messages queued",
        "ERROR: Push notification service down — FCM returned 503",
        "ALERT: Notification storm detected — {count} notifications/sec",
        "ERROR: SMS delivery failure rate at {pct}% — provider degraded",
        "CRITICAL: Notification queue backlog — {count} unprocessed messages",
    ],
    "search-service": [
        "CRITICAL: Elasticsearch cluster RED — {count} shards unassigned",
        "ERROR: Search index corruption detected — reindex required",
        "ALERT: Search latency p99 at {lat}ms — 20x normal baseline",
        "CRITICAL: Out of memory during bulk indexing — heap at 98%",
        "ERROR: Search relevance model failed to load — serving stale results",
    ],
}

IPS = [
    "192.168.1.100", "10.0.0.50", "172.16.0.25", "10.10.5.12",
    "192.168.10.200", "10.0.1.15", "172.20.0.88", "10.50.2.33",
]

SUSPICIOUS_IPS = [
    "45.33.32.156", "185.220.101.34", "91.219.236.13",
    "23.129.64.210", "198.51.100.42", "203.0.113.99",
]


def _fill_template(template: str) -> str:
    """Fill a log message template with random realistic values."""
    replacements = {
        "{uid}": str(random.randint(10000, 99999)),
        "{sid}": f"sess_{random.randint(100000, 999999)}",
        "{client}": f"client_{random.randint(100, 999)}",
        "{service}": random.choice(SERVICES),
        "{lat}": str(random.randint(1, 5000)),
        "{method}": random.choice(["GET", "POST", "PUT", "DELETE"]),
        "{path}": random.choice(["/api/v1/users", "/api/v1/orders", "/api/v1/search", "/health"]),
        "{status}": str(random.choice([200, 201, 204, 301, 400, 404, 500, 503])),
        "{amount}": f"{random.uniform(1.0, 9999.99):.2f}",
        "{txn}": f"txn_{random.randint(1000000, 9999999)}",
        "{inv}": f"inv_{random.randint(100000, 999999)}",
        "{size}": str(random.randint(10, 5000)),
        "{template}": random.choice(["welcome", "reset_password", "invoice", "alert"]),
        "{device}": f"device_{random.randint(1000, 9999)}",
        "{count}": str(random.randint(10, 10000)),
        "{endpoint}": f"https://hooks.example.com/{random.randint(100, 999)}",
        "{query}": random.choice(["machine learning", "kubernetes error", "payment failed", "user login"]),
        "{hash}": f"{random.randint(100000, 999999):x}",
        "{prefix}": random.choice(["mac", "kub", "pay", "err"]),
        "{ip}": random.choice(SUSPICIOUS_IPS),
        "{pct}": str(random.randint(15, 80)),
    }
    result = template
    for key, value in replacements.items():
        result = result.replace(key, value)
    return result


def generate_logs(
    num_logs: int = 50000,
    anomaly_ratio: float = 0.05,
    start_time: Optional[datetime] = None,
    output_dir: str = "data/raw",
) -> str:
    """
    Generate synthetic log files for the platform.

    Args:
        num_logs: Total number of log lines to generate.
        anomaly_ratio: Fraction of logs that are anomalous (0.0-1.0).
        start_time: Starting timestamp for logs.
        output_dir: Directory to write output files.

    Returns:
        Path to generated log file.
    """
    if start_time is None:
        start_time = datetime.now() - timedelta(days=7)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log_lines: List[str] = []
    csv_rows: List[str] = []
    csv_rows.append("timestamp,level,service,source_ip,message,is_anomaly")

    current_time = start_time
    time_step = timedelta(days=7) / num_logs

    num_anomalies = int(num_logs * anomaly_ratio)
    anomaly_indices = set(random.sample(range(num_logs), num_anomalies))

    for i in range(num_logs):
        # Add natural time variance
        jitter = timedelta(seconds=random.uniform(-2.0, 2.0))
        log_time = current_time + jitter

        is_anomaly = i in anomaly_indices
        service = random.choice(SERVICES)

        if is_anomaly:
            level = random.choices(LOG_LEVELS, weights=LEVEL_WEIGHTS_ANOMALOUS, k=1)[0]
            templates = ANOMALOUS_MESSAGES.get(service, ANOMALOUS_MESSAGES["auth-service"])
            ip = random.choice(SUSPICIOUS_IPS)
        else:
            level = random.choices(LOG_LEVELS, weights=LEVEL_WEIGHTS_NORMAL, k=1)[0]
            templates = NORMAL_MESSAGES.get(service, NORMAL_MESSAGES["auth-service"])
            ip = random.choice(IPS)

        message = _fill_template(random.choice(templates))
        timestamp_str = log_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # App-style log line
        log_line = f"{timestamp_str} [{level}] {service}: {message} (src={ip})"
        log_lines.append(log_line)

        # CSV row
        csv_message = message.replace('"', '""')
        csv_rows.append(
            f'{timestamp_str},{level},{service},{ip},"{csv_message}",{int(is_anomaly)}'
        )

        current_time += time_step

    # Write .log file
    log_file = output_path / "system_logs.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    # Write .csv file
    csv_file = output_path / "system_logs.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_rows))

    # Write a small JSON-lines sample
    jsonl_file = output_path / "system_logs.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for i in range(min(5000, num_logs)):
            entry = {
                "timestamp": (start_time + time_step * i).isoformat(),
                "level": log_lines[i].split("] ")[0].split("[")[-1] if "[" in log_lines[i] else "INFO",
                "service": random.choice(SERVICES),
                "message": log_lines[i].split(": ", 1)[-1] if ": " in log_lines[i] else log_lines[i],
                "source_ip": random.choice(IPS + SUSPICIOUS_IPS),
                "is_anomaly": i in anomaly_indices,
            }
            f.write(json.dumps(entry) + "\n")

    print(f"✅ Generated {num_logs} logs ({num_anomalies} anomalies)")
    print(f"   📄 Log file:  {log_file}")
    print(f"   📊 CSV file:  {csv_file}")
    print(f"   📋 JSONL file: {jsonl_file}")

    return str(log_file)


if __name__ == "__main__":
    generate_logs(num_logs=50000, anomaly_ratio=0.05)
