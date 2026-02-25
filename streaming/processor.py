"""
Real-Time Streaming Anomaly Detection Simulator.

Simulates Kafka-style streaming infrastructure:
- Log Producer: generates streaming log events
- Log Consumer: processes and scores logs in near real-time
- Alert Queue: pushes anomalies to an alert system

In production, replace with:
- Apache Kafka / AWS Kinesis / Google Pub/Sub
- Apache Flink / Spark Structured Streaming
- PagerDuty / OpsGenie for alerting
"""

import json
import queue
import random
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class LogProducer:
    """
    Simulates a Kafka producer generating streaming log events.

    Generates realistic log traffic with configurable:
    - Log rate (events per second)
    - Service distribution
    - Anomaly injection rate
    """

    SERVICES = [
        "auth-service", "api-gateway", "payment-service",
        "user-service", "notification-service", "search-service",
    ]

    NORMAL_TEMPLATES = [
        "Request processed successfully in {lat}ms",
        "User {uid} authenticated via {method}",
        "Health check passed for {service}",
        "Cache hit ratio: {pct}%",
        "Connection pool: {count} active",
        "Batch job completed: {count} records processed",
    ]

    ANOMALY_TEMPLATES = [
        "CRITICAL: Service {service} unresponsive — timeout after 30s",
        "ALERT: Brute force attack from IP {ip} — {count} attempts",
        "ERROR: Database connection pool exhausted — {count} pending",
        "CRITICAL: Memory usage at {pct}% — OOM imminent",
        "ALERT: Suspicious access pattern from {ip}",
        "ERROR: Payment gateway timeout — transaction stuck",
    ]

    def __init__(
        self,
        output_queue: queue.Queue,
        rate_per_second: float = 100.0,
        anomaly_rate: float = 0.05,
    ) -> None:
        self.output_queue = output_queue
        self.rate_per_second = rate_per_second
        self.anomaly_rate = anomaly_rate
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._produced_count = 0
        logger.info(
            f"LogProducer initialized: rate={rate_per_second}/s, "
            f"anomaly_rate={anomaly_rate}"
        )

    def start(self) -> None:
        """Start producing log events in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._produce_loop, daemon=True)
        self._thread.start()
        logger.info("LogProducer started")

    def stop(self) -> None:
        """Stop the producer."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(f"LogProducer stopped. Total produced: {self._produced_count}")

    def _produce_loop(self) -> None:
        """Main production loop."""
        interval = 1.0 / self.rate_per_second

        while self._running:
            try:
                event = self._generate_event()
                self.output_queue.put(event, timeout=1)
                self._produced_count += 1
                time.sleep(interval)
            except queue.Full:
                logger.warning("Output queue full, dropping event")
            except Exception as e:
                logger.error(f"Producer error: {e}")

    def _generate_event(self) -> Dict[str, Any]:
        """Generate a single log event."""
        is_anomaly = random.random() < self.anomaly_rate
        service = random.choice(self.SERVICES)

        templates = self.ANOMALY_TEMPLATES if is_anomaly else self.NORMAL_TEMPLATES
        template = random.choice(templates)

        message = template.format(
            lat=random.randint(1, 5000),
            uid=random.randint(10000, 99999),
            method=random.choice(["password", "oauth", "mfa", "api_key"]),
            service=service,
            pct=random.randint(50, 99),
            count=random.randint(10, 10000),
            ip=f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,255)}",
        )

        level = random.choice(["ERROR", "CRITICAL", "WARNING"]) if is_anomaly else random.choice(["INFO", "DEBUG"])

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "service": service,
            "source_ip": f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,255)}",
            "message": message,
            "is_anomaly_injected": is_anomaly,
        }

    @property
    def produced_count(self) -> int:
        return self._produced_count


class LogConsumer:
    """
    Simulates a Kafka consumer that processes and scores streaming logs.

    Accumulates logs in micro-batches, scores them, and pushes
    detected anomalies to an alert queue.
    """

    def __init__(
        self,
        input_queue: queue.Queue,
        alert_queue: queue.Queue,
        scoring_fn: Optional[Callable] = None,
        batch_window_seconds: float = 5.0,
        batch_size: int = 100,
    ) -> None:
        self.input_queue = input_queue
        self.alert_queue = alert_queue
        self.scoring_fn = scoring_fn
        self.batch_window_seconds = batch_window_seconds
        self.batch_size = batch_size
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._consumed_count = 0
        self._anomaly_count = 0
        self._batch_buffer: List[Dict[str, Any]] = []
        logger.info("LogConsumer initialized")

    def start(self) -> None:
        """Start consuming and scoring log events."""
        self._running = True
        self._thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._thread.start()
        logger.info("LogConsumer started")

    def stop(self) -> None:
        """Stop the consumer."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(
            f"LogConsumer stopped. Consumed: {self._consumed_count}, "
            f"Anomalies: {self._anomaly_count}"
        )

    def _consume_loop(self) -> None:
        """Main consumption loop with micro-batching."""
        last_flush = time.time()

        while self._running:
            try:
                # Drain queue into buffer
                try:
                    event = self.input_queue.get(timeout=0.5)
                    self._batch_buffer.append(event)
                    self._consumed_count += 1
                except queue.Empty:
                    pass

                # Flush batch if window expired or batch is full
                elapsed = time.time() - last_flush
                if (
                    len(self._batch_buffer) >= self.batch_size
                    or elapsed >= self.batch_window_seconds
                ) and self._batch_buffer:
                    self._process_batch(list(self._batch_buffer))
                    self._batch_buffer.clear()
                    last_flush = time.time()

            except Exception as e:
                logger.error(f"Consumer error: {e}")

    def _process_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Process and score a micro-batch of log events."""
        start = time.perf_counter()

        if self.scoring_fn:
            try:
                scores = self.scoring_fn(batch)
                for event, score in zip(batch, scores):
                    event["anomaly_score"] = float(score)
                    if score > 0.5:  # Anomaly threshold
                        event["is_anomaly_detected"] = True
                        self._anomaly_count += 1
                        self.alert_queue.put(event)
                    else:
                        event["is_anomaly_detected"] = False
            except Exception as e:
                logger.error(f"Scoring failed for batch: {e}")
                # Fallback: use injected labels
                for event in batch:
                    if event.get("is_anomaly_injected", False):
                        event["anomaly_score"] = random.uniform(0.7, 1.0)
                        event["is_anomaly_detected"] = True
                        self._anomaly_count += 1
                        self.alert_queue.put(event)
        else:
            # No scoring function — use heuristic
            for event in batch:
                level = event.get("level", "INFO")
                is_anomaly = level in ("ERROR", "CRITICAL") and random.random() < 0.3
                event["anomaly_score"] = random.uniform(0.6, 0.95) if is_anomaly else random.uniform(0.01, 0.4)
                event["is_anomaly_detected"] = is_anomaly
                if is_anomaly:
                    self._anomaly_count += 1
                    self.alert_queue.put(event)

        duration = (time.perf_counter() - start) * 1000
        if len(batch) > 0:
            logger.info(
                f"Batch processed: {len(batch)} logs, "
                f"{self._anomaly_count} anomalies total ({duration:.1f}ms)"
            )

    @property
    def consumed_count(self) -> int:
        return self._consumed_count

    @property
    def anomaly_count(self) -> int:
        return self._anomaly_count


class StreamingPipeline:
    """
    End-to-end streaming anomaly detection pipeline.

    Connects Producer → Consumer → Alert Queue.
    """

    def __init__(
        self,
        rate_per_second: float = 50.0,
        anomaly_rate: float = 0.05,
        scoring_fn: Optional[Callable] = None,
    ) -> None:
        self.log_queue: queue.Queue = queue.Queue(maxsize=10000)
        self.alert_queue: queue.Queue = queue.Queue(maxsize=5000)

        self.producer = LogProducer(
            output_queue=self.log_queue,
            rate_per_second=rate_per_second,
            anomaly_rate=anomaly_rate,
        )
        self.consumer = LogConsumer(
            input_queue=self.log_queue,
            alert_queue=self.alert_queue,
            scoring_fn=scoring_fn,
        )

    def start(self) -> None:
        """Start the streaming pipeline."""
        logger.info("🚀 Starting streaming pipeline...")
        self.consumer.start()
        self.producer.start()

    def stop(self) -> None:
        """Stop the streaming pipeline gracefully."""
        logger.info("Stopping streaming pipeline...")
        self.producer.stop()
        time.sleep(2)  # Let consumer drain
        self.consumer.stop()

    def get_alerts(self, max_alerts: int = 100) -> List[Dict[str, Any]]:
        """Drain the alert queue."""
        alerts = []
        while not self.alert_queue.empty() and len(alerts) < max_alerts:
            try:
                alerts.append(self.alert_queue.get_nowait())
            except queue.Empty:
                break
        return alerts

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "produced": self.producer.produced_count,
            "consumed": self.consumer.consumed_count,
            "anomalies_detected": self.consumer.anomaly_count,
            "queue_size": self.log_queue.qsize(),
            "alert_queue_size": self.alert_queue.qsize(),
        }


def run_streaming_demo(duration_seconds: int = 30) -> None:
    """Run a streaming demo for a specified duration."""
    pipeline = StreamingPipeline(rate_per_second=20, anomaly_rate=0.05)
    pipeline.start()

    try:
        for i in range(duration_seconds):
            time.sleep(1)
            stats = pipeline.get_stats()
            alerts = pipeline.get_alerts(max_alerts=5)

            print(f"\r⏱  t={i+1}s | "
                  f"Produced: {stats['produced']} | "
                  f"Consumed: {stats['consumed']} | "
                  f"Anomalies: {stats['anomalies_detected']} | "
                  f"Queue: {stats['queue_size']} | "
                  f"Alerts: {len(alerts)}", end="")

            for alert in alerts[:2]:
                print(f"\n  🚨 ALERT: [{alert.get('service')}] {alert.get('message', '')[:80]}")

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        pipeline.stop()
        stats = pipeline.get_stats()
        print(f"\n\n📊 Final Stats:")
        print(f"   Produced: {stats['produced']}")
        print(f"   Consumed: {stats['consumed']}")
        print(f"   Anomalies: {stats['anomalies_detected']}")


if __name__ == "__main__":
    run_streaming_demo(30)
