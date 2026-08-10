from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from dfi.config import load_config

ROOT = Path(__file__).resolve().parents[1]
REVISION = "0f2787f2d87eac5eed8a087d5ecd24277e6255b2"


def _config_text(
    *,
    dataset_path: str = "../data/anchors.jsonl",
    cache_root: str = "/content/drive/MyDrive/DFI/cache-v1",
) -> str:
    return f'''config_version = 1
seed = 7
output_root = "../runs"

[dataset]
path = "{dataset_path}"

[model]
repository = "GSAI-ML/LLaDA-8B-Base"
revision = "{REVISION}"
tokenizer_revision = "{REVISION}"
remote_code_revision = "{REVISION}"

[scoring]
protocol = "analytic-v1"
arms = ["prior"]
mask_policy = "fixed-v1"
n_masks = 32
temperature = 1.0
prompt_protocol = "claim-prefix-v1"

[runtime]
dtype = "bfloat16"
max_batch_tokens = 8192
max_batch_rows = 64
partition_requests = 4096

[cache]
backend = "google_drive"
root = "{cache_root}"
'''


def _write_config(tmp_path: Path, text: str | None = None) -> Path:
    config_directory = tmp_path / "configs"
    data_directory = tmp_path / "data"
    config_directory.mkdir(parents=True)
    data_directory.mkdir()
    (data_directory / "anchors.jsonl").write_text("{}\n", encoding="utf-8")
    path = config_directory / "run.toml"
    path.write_text(text if text is not None else _config_text(), encoding="utf-8")
    return path


def test_checked_in_a100_config_is_complete_and_resolved() -> None:
    config = load_config(ROOT / "configs" / "a100-smoke.toml")

    assert config.dataset.path == (ROOT / "data" / "fixtures" / "anchors.jsonl").resolve()
    assert config.output_root == (ROOT / "runs").resolve()
    assert config.cache.root == Path("/content/drive/MyDrive/DFI/cache-v1")
    assert config.model.revision == REVISION
    assert config.model.tokenizer_revision == REVISION
    assert config.model.remote_code_revision == REVISION
    assert config.scoring.arms == ("prior",)
    assert config.scoring.temperature == 1.0

    throughput = load_config(ROOT / "configs" / "a100-throughput.toml")
    assert throughput.dataset.path == config.dataset.path
    assert throughput.model == config.model
    assert throughput.scoring.n_masks == 512
    assert throughput.runtime.partition_requests == 4096
    assert throughput.cache == config.cache


def test_config_is_frozen_and_receipt_serialization_contains_no_paths(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))

    with pytest.raises(FrozenInstanceError):
        config.seed = 8  # type: ignore[misc]
    receipt = config.to_receipt_dict()
    assert json.loads(json.dumps(receipt)) == receipt
    assert receipt["dataset"]["path"] == str((tmp_path / "data" / "anchors.jsonl").resolve())
    assert receipt["output_root"] == str((tmp_path / "runs").resolve())
    assert receipt["scoring"]["prompt_protocol"] == "claim-prefix-v1"


@pytest.mark.parametrize(
    ("marker", "replacement"),
    [
        ("[dataset]", 'unexpected = "top"\n\n[dataset]'),
        ('path = "../data/anchors.jsonl"', 'path = "../data/anchors.jsonl"\nextra = true'),
        (
            'repository = "GSAI-ML/LLaDA-8B-Base"',
            'repository = "GSAI-ML/LLaDA-8B-Base"\nextra = true',
        ),
        ('protocol = "analytic-v1"', 'protocol = "analytic-v1"\nextra = true'),
        ('dtype = "bfloat16"', 'dtype = "bfloat16"\nextra = true'),
        ('backend = "google_drive"', 'backend = "google_drive"\nextra = true'),
    ],
    ids=["top", "dataset", "model", "scoring", "runtime", "cache"],
)
def test_unknown_keys_are_rejected_at_every_level(
    tmp_path: Path, marker: str, replacement: str
) -> None:
    path = _write_config(tmp_path, _config_text().replace(marker, replacement, 1))
    with pytest.raises(ValueError, match="invalid keys"):
        load_config(path)


@pytest.mark.parametrize(
    "line",
    [
        "config_version = 1\n",
        'path = "../data/anchors.jsonl"\n',
        f'revision = "{REVISION}"\n',
        "n_masks = 32\n",
        'dtype = "bfloat16"\n',
        'backend = "google_drive"\n',
    ],
    ids=["top", "dataset", "model", "scoring", "runtime", "cache"],
)
def test_missing_keys_are_rejected_at_every_level(tmp_path: Path, line: str) -> None:
    path = _write_config(tmp_path, _config_text().replace(line, "", 1))
    with pytest.raises(ValueError, match="invalid keys"):
        load_config(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("config_version = 1", "config_version = 2", "config_version"),
        ("seed = 7", "seed = -1", "seed"),
        (
            'repository = "GSAI-ML/LLaDA-8B-Base"',
            'repository = "GSAI-ML/LLaDA-8B-Instruct"',
            "model.repository",
        ),
        (f'revision = "{REVISION}"', 'revision = "main"', "model.revision"),
        (
            f'tokenizer_revision = "{REVISION}"',
            f'tokenizer_revision = "{REVISION.upper()}"',
            "model.tokenizer_revision",
        ),
        (
            f'remote_code_revision = "{REVISION}"',
            'remote_code_revision = "0123456789abcdef"',
            "model.remote_code_revision",
        ),
        ('protocol = "analytic-v1"', 'protocol = "sampled-v1"', "scoring.protocol"),
        ('arms = ["prior"]', 'arms = ["supplied_evidence"]', "scoring.arms"),
        (
            'arms = ["prior"]',
            'arms = ["supplied_evidence", "prior"]',
            "scoring.arms",
        ),
        ('mask_policy = "fixed-v1"', 'mask_policy = "adaptive-v1"', "mask_policy"),
        ("n_masks = 32", "n_masks = 0", "n_masks"),
        ("temperature = 1.0", "temperature = 1", "temperature"),
        ("temperature = 1.0", "temperature = 0.5", "temperature"),
        (
            'prompt_protocol = "claim-prefix-v1"',
            'prompt_protocol = "chat-v1"',
            "prompt_protocol",
        ),
        ('dtype = "bfloat16"', 'dtype = "float16"', "runtime.dtype"),
        ("max_batch_tokens = 8192", "max_batch_tokens = 0", "max_batch_tokens"),
        ("max_batch_rows = 64", "max_batch_rows = -1", "max_batch_rows"),
        ("partition_requests = 4096", "partition_requests = 0", "partition_requests"),
        ('backend = "google_drive"', 'backend = "local"', "cache.backend"),
        (
            'root = "/content/drive/MyDrive/DFI/cache-v1"',
            'root = "relative/cache"',
            "cache.root",
        ),
    ],
)
def test_invalid_values_fail_closed(tmp_path: Path, old: str, new: str, message: str) -> None:
    path = _write_config(tmp_path, _config_text().replace(old, new, 1))
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_two_supported_arm_shapes_are_the_only_accepted_values(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        _config_text().replace('arms = ["prior"]', 'arms = ["prior", "supplied_evidence"]', 1),
    )
    assert load_config(path).scoring.arms == ("prior", "supplied_evidence")


def test_dataset_must_exist_but_cache_root_need_not(tmp_path: Path) -> None:
    absent_cache = tmp_path / "mounted-drive" / "cache-v1"
    path = _write_config(tmp_path, _config_text(cache_root=str(absent_cache)))
    config = load_config(path)
    assert config.cache.root == absent_cache
    assert not absent_cache.exists()

    missing = _write_config(tmp_path / "other", _config_text(dataset_path="missing.jsonl"))
    with pytest.raises(ValueError, match=r"dataset\.path does not name an existing file"):
        load_config(missing)


def test_malformed_toml_and_missing_config_are_concise_errors(tmp_path: Path) -> None:
    malformed = tmp_path / "bad.toml"
    malformed.write_text("config_version = [", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid TOML"):
        load_config(malformed)
    with pytest.raises(ValueError, match="does not exist"):
        load_config(tmp_path / "missing.toml")
