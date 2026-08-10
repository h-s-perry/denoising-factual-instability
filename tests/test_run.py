from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import dfi.cli as cli
import dfi.pipeline as pipeline
from dfi.cli import main
from dfi.config import DFIConfig, load_config
from dfi.data import load_jsonl
from dfi.llada import (
    AnalyticResult,
    LLaDAModelSpec,
    ParityMismatchError,
    WorkRequest,
    plan_length_buckets,
)
from dfi.masking import MaskChoice, stable_work_id
from dfi.pipeline import (
    build_inference_plan,
    evaluate_run_bundle,
    execute_inference_requests,
    read_parquet_rows,
    run_experiment,
)
from dfi.scoring import score_marginals

ROOT = Path(__file__).resolve().parents[1]
REVISION = "0f2787f2d87eac5eed8a087d5ecd24277e6255b2"


def _write_config(
    tmp_path: Path,
    *,
    partition_requests: int = 4,
    n_masks: int = 2,
) -> tuple[Path, DFIConfig]:
    path = tmp_path / "experiment.toml"
    text = f'''config_version = 1
seed = 7
output_root = "{tmp_path / "runs"}"

[dataset]
path = "{ROOT / "data" / "fixtures" / "anchors.jsonl"}"

[model]
repository = "GSAI-ML/LLaDA-8B-Base"
revision = "{REVISION}"
tokenizer_revision = "{REVISION}"
remote_code_revision = "{REVISION}"

[scoring]
protocol = "analytic-v1"
arms = ["prior"]
mask_policy = "fixed-v1"
n_masks = {n_masks}
temperature = 1.0
prompt_protocol = "claim-prefix-v1"

[runtime]
dtype = "bfloat16"
max_batch_tokens = 64
max_batch_rows = 4
partition_requests = {partition_requests}

[cache]
backend = "google_drive"
root = "{tmp_path / "drive-cache"}"
'''
    path.write_text(text, encoding="utf-8")
    return path, load_config(path)


class FakeOOM(RuntimeError):
    pass


class FakeBackend:
    def __init__(
        self,
        *,
        fail_on_call: int | None = None,
        oom_once: bool = False,
        oom_always: bool = False,
    ) -> None:
        self.spec = LLaDAModelSpec(
            repository="GSAI-ML/LLaDA-8B-Base",
            revision=REVISION,
            tokenizer_revision=REVISION,
            remote_code_revision=REVISION,
        )
        self.fail_on_call = fail_on_call
        self.oom_once = oom_once
        self.oom_always = oom_always
        self.calls: list[tuple[str, ...]] = []
        self.reset_count = 0

    def prepare_request(
        self,
        *,
        example_id: str,
        claim: str,
        evidence: str | None,
        arm: str,
        choice: MaskChoice,
        mask_policy: str,
    ) -> WorkRequest:
        digest = hashlib.sha256(f"{claim}\0{evidence}\0{choice.word_indices}".encode()).digest()
        length = 3 + digest[0] % 4
        target = int(digest[1] % 3)
        values = [10 + int(value % 50) for value in digest[:length]]
        values[-1] = 126336
        masked_positions = (length - 1,)
        input_ids = tuple(values)
        attention = (1,) * length
        work_id = stable_work_id(
            scoring_protocol="analytic-v1",
            model_repository=self.spec.repository,
            model_revision=self.spec.revision,
            tokenizer_revision=self.spec.tokenizer_revision,
            remote_code_revision=self.spec.remote_code_revision,
            arm=arm,
            prompt_protocol="claim-prefix-v1",
            mask_policy=mask_policy,
            mask_rate=choice.mask_rate,
            input_ids=input_ids,
            attention_mask=attention,
            masked_positions=masked_positions,
            target_ids=(target,),
            temperature=1.0,
            dtype="bfloat16",
            inference_backend="llada-full-logits-v1",
        )
        return WorkRequest(
            work_id=work_id,
            example_id=example_id,
            arm=arm,
            mask_index=choice.mask_index,
            mask_rate=choice.mask_rate,
            word_indices=choice.word_indices,
            input_ids=input_ids,
            attention_mask=attention,
            masked_positions=masked_positions,
            target_ids=(target,),
        )

    def infer_batch(self, requests: tuple[WorkRequest, ...]) -> tuple[AnalyticResult, ...]:
        self.calls.append(tuple(request.work_id for request in requests))
        call_number = len(self.calls)
        if self.fail_on_call == call_number:
            raise RuntimeError("injected interruption")
        if self.oom_always or (self.oom_once and call_number == 1):
            raise FakeOOM("fake CUDA OOM")
        results: list[AnalyticResult] = []
        for request in requests:
            offset = int(request.work_id[:2], 16) / 2550.0
            probabilities = np.asarray([[0.70 - offset, 0.20 + offset, 0.10]])
            stats = score_marginals(probabilities, request.target_ids)
            results.append(
                AnalyticResult(
                    work_id=request.work_id,
                    masked_positions=request.masked_positions,
                    target_ids=tuple(int(value) for value in stats.target_ids),
                    top_ids=tuple(int(value) for value in stats.top_ids),
                    top_alternative_ids=tuple(int(value) for value in stats.top_alternative_ids),
                    ce=tuple(float(value) for value in stats.ce),
                    entropy=tuple(float(value) for value in stats.entropy),
                    collision_probability=tuple(
                        float(value) for value in stats.collision_probability
                    ),
                    collision_entropy=tuple(float(value) for value in stats.collision_entropy),
                    delta=tuple(float(value) for value in stats.delta),
                    swap_llr=tuple(float(value) for value in stats.swap_llr),
                    expected_drift=tuple(float(value) for value in stats.expected_drift),
                    expected_dispersion=tuple(float(value) for value in stats.expected_dispersion),
                    top_mismatch=tuple(float(value) for value in stats.top_mismatch),
                )
            )
        return tuple(results)

    def infer_scalar(self, request: WorkRequest) -> AnalyticResult:
        return self.infer_batch((request,))[0]

    def infer_global(
        self,
        requests: tuple[WorkRequest, ...],
        *,
        max_batch_tokens: int,
        max_batch_rows: int,
    ) -> tuple[AnalyticResult, ...]:
        by_id: dict[str, AnalyticResult] = {}
        for batch in plan_length_buckets(
            requests,
            max_batch_tokens=max_batch_tokens,
            max_batch_rows=max_batch_rows,
        ):
            for result in self.infer_batch(batch):
                by_id[result.work_id] = result
        return tuple(by_id[request.work_id] for request in requests)

    @staticmethod
    def is_oom_error(error: BaseException) -> bool:
        return isinstance(error, FakeOOM)

    def reset_peak_vram(self) -> None:
        self.reset_count += 1

    @staticmethod
    def peak_vram_bytes() -> int:
        return 123_456

    @staticmethod
    def runtime_environment() -> dict[str, Any]:
        return {
            "torch": "fake-torch",
            "transformers": "fake-transformers",
            "cuda_runtime": "fake-cuda",
            "cuda_driver": "fake-driver",
            "device": "cuda",
            "device_name": "Fake A100",
            "device_capability": [8, 0],
            "dtype": "bfloat16",
            "model_parameter_dtype": "bfloat16",
        }


def _receipt(run_directory: Path) -> dict[str, Any]:
    return json.loads((run_directory / "run.json").read_text(encoding="utf-8"))


def test_archive_marker_recovers_commit_without_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "1" * 40
    marker = tmp_path / ".git_archival.txt"
    marker.write_text(f"commit: {commit}\n", encoding="utf-8")
    anchor = tmp_path / "configs" / "experiment.toml"
    anchor.parent.mkdir()
    anchor.write_text("config_version = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=args, returncode=1),
    )

    assert pipeline._repository_state(anchor) == {
        "available": True,
        "commit": commit,
        "dirty": None,
    }


def test_plan_is_invariant_to_example_order_and_separates_logical_counts(
    tmp_path: Path,
) -> None:
    _, config = _write_config(tmp_path)
    _, examples = load_jsonl(config.dataset.path)
    backend = FakeBackend()
    forward = build_inference_plan(config, examples, backend)
    reverse = build_inference_plan(config, list(reversed(examples)), backend)

    assert forward.plan_hash == reverse.plan_hash
    assert forward.execution_order == reverse.execution_order
    assert forward.logical_requests == reverse.logical_requests
    assert len(forward.logical_requests) == len(examples) * config.scoring.n_masks
    assert len(forward.physical_requests) <= len(forward.logical_requests)
    assert set().union(*forward.partitions.values()) == forward.expected_work_ids


def test_cold_run_then_fresh_local_warm_run_uses_zero_model_forwards(tmp_path: Path) -> None:
    config_path, config = _write_config(tmp_path)
    cold_backend = FakeBackend()
    cold = run_experiment(
        config,
        config_path=config_path,
        backend=cold_backend,
        allow_local_cache_for_testing=True,
    )
    cold_rows = read_parquet_rows(cold / "results.parquet")
    cold_receipt = _receipt(cold)

    assert set(path.name for path in cold.iterdir()) == {"results.parquet", "run.json"}
    assert cold_receipt["requests"]["computed_requests"] == 16
    assert cold_receipt["requests"]["forwards"] > 0
    assert cold_receipt["cache"]["hits"] == 0
    assert cold_receipt["interpretation_allowed"] is False

    warm_backend = FakeBackend()
    warm = run_experiment(
        config,
        config_path=config_path,
        backend=warm_backend,
        allow_local_cache_for_testing=True,
    )
    warm_receipt = _receipt(warm)
    assert warm_backend.calls == []
    assert warm_receipt["requests"]["computed_requests"] == 0
    assert warm_receipt["requests"]["forwards"] == 0
    assert warm_receipt["cache"]["hits"] == 16
    assert warm_receipt["cache"]["misses"] == 0
    assert warm_receipt["cache"]["reused_creation_environments"]["drive"] == [
        cold_receipt["environment"]
    ]
    assert warm_receipt["cache"]["reused_creation_environments"]["local_resume"] == []
    assert warm_receipt["results"]["sha256"] == cold_receipt["results"]["sha256"]
    assert read_parquet_rows(warm / "results.parquet") == cold_rows


def test_completed_run_recovery_only_cleans_crash_leftovers(tmp_path: Path) -> None:
    config_path, config = _write_config(tmp_path)
    completed = run_experiment(
        config,
        config_path=config_path,
        backend=FakeBackend(),
        allow_local_cache_for_testing=True,
    )
    original_receipt = (completed / "run.json").read_bytes()
    original_results = (completed / "results.parquet").read_bytes()
    (completed / ".partial").mkdir()
    (completed / ".partial" / "leftover.parquet").write_bytes(b"crash-leftover")
    (completed / "run.partial.json").write_text("{}", encoding="utf-8")
    staging = completed.parent / f".{completed.name}.finalizing-crash"
    staging.mkdir()
    (staging / "leftover").write_bytes(b"crash-leftover")

    unused_backend = FakeBackend(fail_on_call=1)
    recovered = run_experiment(
        config,
        config_path=config_path,
        backend=unused_backend,
        resume_directory=completed,
        allow_local_cache_for_testing=True,
    )

    assert recovered == completed
    assert unused_backend.calls == []
    assert (completed / "run.json").read_bytes() == original_receipt
    assert (completed / "results.parquet").read_bytes() == original_results
    assert set(path.name for path in completed.iterdir()) == {"results.parquet", "run.json"}
    assert not staging.exists()


def test_evaluate_run_bundle_recomputes_and_rejects_changed_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path, config = _write_config(tmp_path)
    completed = run_experiment(
        config,
        config_path=config_path,
        backend=FakeBackend(),
        allow_local_cache_for_testing=True,
    )
    receipt = _receipt(completed)
    assert evaluate_run_bundle(completed) == receipt["evaluation"]
    assert main(["evaluate", str(completed)]) == 0
    assert json.loads(capsys.readouterr().out) == receipt["evaluation"]

    receipt["evaluation"]["n_mask_rows"] += 1
    (completed / "run.json").write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stored evaluation does not match"):
        evaluate_run_bundle(completed)


def test_corrupt_drive_partition_is_recomputed_in_isolation(tmp_path: Path) -> None:
    config_path, config = _write_config(tmp_path)
    cold = run_experiment(
        config,
        config_path=config_path,
        backend=FakeBackend(),
        allow_local_cache_for_testing=True,
    )
    cache_directory = Path(_receipt(cold)["cache"]["plan_directory"])
    manifest = json.loads((cache_directory / "cache.json").read_text(encoding="utf-8"))
    corrupt = cache_directory / manifest["shards"][0]["name"]
    corrupt.write_bytes(corrupt.read_bytes()[:-8])

    repair_backend = FakeBackend()
    repaired = run_experiment(
        config,
        config_path=config_path,
        backend=repair_backend,
        allow_local_cache_for_testing=True,
    )
    receipt = _receipt(repaired)
    assert receipt["cache"]["validation_failures"] == 1
    assert receipt["cache"]["hits"] == 12
    assert receipt["cache"]["misses"] == 4
    assert receipt["requests"]["computed_requests"] == 4
    assert sum(len(batch) for batch in repair_backend.calls) == 4


def test_runner_executes_and_binds_requested_parity_gate(tmp_path: Path) -> None:
    config_path, config = _write_config(tmp_path, n_masks=1)
    completed = run_experiment(
        config,
        config_path=config_path,
        backend=FakeBackend(),
        run_parity=True,
        allow_local_cache_for_testing=True,
    )
    receipt = _receipt(completed)
    parity = receipt["acceptance"]["scalar_batch_parity"]
    assert parity["status"] == "passed"
    assert parity["plan_hash"] == receipt["plan"]["hash"]
    assert parity["report"]["passed"] is True
    assert parity["report"]["request_count"] == 8
    assert parity["report"]["selection"]["logical_request_count"] == 8
    assert parity["report"]["selection"]["family_count"] == 4
    assert parity["report"]["selection"]["mask_index"] == 0
    assert parity["report"]["aggregate_checks"]["passed"] is True
    assert len(parity["report"]["expected_work_ids"]) == receipt["requests"]["unique_requests"]
    assert len(set(parity["report"]["expected_work_ids"])) == len(
        parity["report"]["expected_work_ids"]
    )
    assert receipt["requests"]["acceptance_forwards"] > 0
    assert receipt["requests"]["forwards"] == (
        receipt["requests"]["inference_forwards"] + receipt["requests"]["acceptance_forwards"]
    )


def test_malformed_manifest_and_corrupt_orphan_are_rejected_and_rebuilt(
    tmp_path: Path,
) -> None:
    config_path, config = _write_config(tmp_path)
    cold = run_experiment(
        config,
        config_path=config_path,
        backend=FakeBackend(),
        allow_local_cache_for_testing=True,
    )
    cache_directory = Path(_receipt(cold)["cache"]["plan_directory"])
    manifest_path = cache_directory / "cache.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    orphan = cache_directory / manifest["shards"][0]["name"]
    orphan.write_bytes(orphan.read_bytes()[:-8])
    malformed_bytes = b'{"truncated":'
    manifest_path.write_bytes(malformed_bytes)

    rebuilt = run_experiment(
        config,
        config_path=config_path,
        backend=FakeBackend(),
        allow_local_cache_for_testing=True,
    )
    receipt = _receipt(rebuilt)
    assert receipt["cache"]["validation_failures"] == 1
    assert receipt["cache"]["rejected_manifest"] == {
        "sha256": hashlib.sha256(malformed_bytes).hexdigest(),
        "size_bytes": len(malformed_bytes),
        "reason": f"cache manifest is missing or malformed: {manifest_path}",
    }
    assert receipt["requests"]["computed_requests"] == 16
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["generation"] == 4


def test_interrupted_run_resumes_exact_local_partitions(tmp_path: Path) -> None:
    config_path, config = _write_config(tmp_path)
    failing = FakeBackend(fail_on_call=2)
    with pytest.raises(RuntimeError, match="injected interruption"):
        run_experiment(
            config,
            config_path=config_path,
            backend=failing,
            allow_local_cache_for_testing=True,
        )
    partial_runs = list(config.output_root.iterdir())
    assert len(partial_runs) == 1
    partial = partial_runs[0]
    state = json.loads((partial / "run.partial.json").read_text(encoding="utf-8"))
    assert len(state["completed_partitions"]) == 1

    resumed_backend = FakeBackend()
    completed = run_experiment(
        config,
        config_path=config_path,
        backend=resumed_backend,
        resume_directory=partial,
        allow_local_cache_for_testing=True,
    )
    receipt = _receipt(completed)
    assert completed == partial
    assert set(path.name for path in completed.iterdir()) == {"results.parquet", "run.json"}
    assert receipt["resume"]["requested"] is True
    assert receipt["resume"]["resume_count"] == 1
    assert receipt["requests"]["completed_requests"] == 16
    assert receipt["requests"]["computed_requests"] == 16
    assert receipt["requests"]["inference_forwards"] == 5


def test_runner_halves_once_on_oom_and_fails_if_oom_recurs() -> None:
    backend = FakeBackend(oom_once=True)
    requests = tuple(
        backend.prepare_request(
            example_id=f"example-{index}",
            claim=f"Claim {index}",
            evidence=None,
            arm="prior",
            choice=MaskChoice(index, 0.4 + index * 0.01, (0,)),
            mask_policy="fixed-v1",
        )
        for index in range(4)
    )
    results, telemetry = execute_inference_requests(
        backend,
        requests,
        max_batch_tokens=64,
        max_batch_rows=4,
    )
    assert len(results) == 4
    assert telemetry.oom_fallbacks == 1
    assert telemetry.fallback_batch_rows == 2
    assert telemetry.forwards == 3

    always = FakeBackend(oom_always=True)
    with pytest.raises(RuntimeError, match="single permitted"):
        execute_inference_requests(
            always,
            requests,
            max_batch_tokens=64,
            max_batch_rows=4,
        )


def test_cli_summarizes_parity_failure_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path, _ = _write_config(tmp_path)
    report = {
        "failure_count": 107,
        "failures": ["numeric mismatch"],
        "exact_checks": {"work_ids": True},
        "numeric_checks": {"max_absolute_error": {"ce": 0.07}},
        "batch_plan": {"batch_count": 1},
    }

    def fail(*args: Any, **kwargs: Any) -> Path:
        del args, kwargs
        raise ParityMismatchError(report)

    monkeypatch.setattr(cli, "run_experiment", fail)
    assert cli.main(["run", str(config_path), "--parity"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "scalar/global parity gate failed" in captured.err
    assert '"failed_comparisons": 107' in captured.err
    assert "Traceback" not in captured.err
