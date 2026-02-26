"""
Pipeline Orchestrator — DAG-Based Workflow Execution.

Simulates Airflow-style DAG orchestration for the ML pipeline:
- Task dependency management
- Parallel and sequential execution
- Retry logic with backoff
- Pipeline state tracking
- Execution logging and metrics

This is a lightweight simulation of production orchestration.
In production, replace with Airflow, Prefect, or Dagster.
"""

import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from utils.helpers import timer
from utils.logger import get_logger

logger = get_logger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class TaskResult:
    """Result of a task execution."""

    status: TaskStatus
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    retries: int = 0
    started_at: str | None = None
    completed_at: str | None = None


@dataclass
class PipelineTask:
    """A single task in the pipeline DAG."""

    name: str
    callable: Callable
    dependencies: list[str] = field(default_factory=list)
    retry_count: int = 3
    retry_delay: float = 1.0
    timeout: float | None = None
    description: str = ""


class PipelineOrchestrator:
    """
    DAG-based pipeline orchestrator.

    Executes tasks in topological order, respecting dependencies,
    with retry logic and comprehensive execution tracking.

    Usage:
        orchestrator = PipelineOrchestrator("training_pipeline")
        orchestrator.add_task("ingest", ingest_fn)
        orchestrator.add_task("validate", validate_fn, depends_on=["ingest"])
        orchestrator.add_task("train", train_fn, depends_on=["validate"])
        results = orchestrator.run(context={})
    """

    def __init__(self, pipeline_name: str) -> None:
        self.pipeline_name = pipeline_name
        self._tasks: dict[str, PipelineTask] = {}
        self._results: dict[str, TaskResult] = {}
        self._execution_order: list[str] = []
        self._context: dict[str, Any] = {}
        logger.info(f"Pipeline '{pipeline_name}' initialized")

    def add_task(
        self,
        name: str,
        callable: Callable,
        depends_on: list[str] | None = None,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        description: str = "",
    ) -> "PipelineOrchestrator":
        """Add a task to the pipeline DAG."""
        task = PipelineTask(
            name=name,
            callable=callable,
            dependencies=depends_on or [],
            retry_count=retry_count,
            retry_delay=retry_delay,
            description=description,
        )
        self._tasks[name] = task
        return self  # Enable chaining

    @timer
    def run(self, context: dict[str, Any] | None = None) -> dict[str, TaskResult]:
        """
        Execute the pipeline DAG.

        Args:
            context: Shared context dictionary passed between tasks.

        Returns:
            Dictionary mapping task names to their results.
        """
        self._context = context or {}
        self._results = {}

        # Resolve execution order (topological sort)
        self._execution_order = self._topological_sort()

        logger.info(
            f"Starting pipeline '{self.pipeline_name}' with "
            f"{len(self._execution_order)} tasks: {self._execution_order}"
        )

        pipeline_start = time.perf_counter()

        for task_name in self._execution_order:
            task = self._tasks[task_name]

            # Check if dependencies succeeded
            deps_ok = all(
                self._results.get(dep, TaskResult(TaskStatus.FAILED)).status == TaskStatus.SUCCESS
                for dep in task.dependencies
            )

            if not deps_ok:
                logger.warning(f"Skipping task '{task_name}' — dependency failure")
                self._results[task_name] = TaskResult(
                    status=TaskStatus.SKIPPED,
                    error="Dependency failure",
                )
                continue

            # Execute with retry
            self._results[task_name] = self._execute_task(task)

        pipeline_duration = (time.perf_counter() - pipeline_start) * 1000
        success_count = sum(1 for r in self._results.values() if r.status == TaskStatus.SUCCESS)
        logger.info(
            f"Pipeline '{self.pipeline_name}' complete: "
            f"{success_count}/{len(self._results)} tasks succeeded "
            f"({pipeline_duration:.0f}ms)"
        )

        return self._results

    def _execute_task(self, task: PipelineTask) -> TaskResult:
        """Execute a single task with retry logic."""
        for attempt in range(task.retry_count + 1):
            start = time.perf_counter()
            started_at = datetime.now(UTC).isoformat()

            try:
                logger.info(
                    f"[{self.pipeline_name}] Running task '{task.name}' "
                    f"(attempt {attempt + 1}/{task.retry_count + 1})"
                )
                output = task.callable(self._context)
                duration = (time.perf_counter() - start) * 1000

                # Store output in context for downstream tasks
                self._context[f"{task.name}_output"] = output

                return TaskResult(
                    status=TaskStatus.SUCCESS,
                    output=output,
                    duration_ms=duration,
                    retries=attempt,
                    started_at=started_at,
                    completed_at=datetime.now(UTC).isoformat(),
                )

            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                logger.error(f"Task '{task.name}' failed (attempt {attempt + 1}): {e}")
                if attempt < task.retry_count:
                    delay = task.retry_delay * (2**attempt)  # Exponential backoff
                    logger.info(f"Retrying '{task.name}' in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    return TaskResult(
                        status=TaskStatus.FAILED,
                        error=str(e),
                        duration_ms=duration,
                        retries=attempt,
                        started_at=started_at,
                        completed_at=datetime.now(UTC).isoformat(),
                    )

        # Should not reach here, but just in case
        return TaskResult(status=TaskStatus.FAILED, error="Unknown error")

    def _topological_sort(self) -> list[str]:
        """Topological sort of tasks based on dependencies."""
        in_degree: dict[str, int] = defaultdict(int)
        graph: dict[str, list[str]] = defaultdict(list)

        for name, task in self._tasks.items():
            in_degree.setdefault(name, 0)
            for dep in task.dependencies:
                graph[dep].append(name)
                in_degree[name] += 1

        # Start with tasks that have no dependencies
        queue = [n for n, d in in_degree.items() if d == 0]
        result: list[str] = []

        while queue:
            # Sort for deterministic order
            queue.sort()
            node = queue.pop(0)
            result.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self._tasks):
            raise ValueError(f"Circular dependency detected in pipeline '{self.pipeline_name}'")

        return result

    def get_summary(self) -> dict[str, Any]:
        """Get execution summary."""
        return {
            "pipeline": self.pipeline_name,
            "tasks": {
                name: {
                    "status": result.status.value,
                    "duration_ms": result.duration_ms,
                    "retries": result.retries,
                    "error": result.error,
                }
                for name, result in self._results.items()
            },
        }
