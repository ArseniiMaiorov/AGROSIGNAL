from pathlib import Path

from training.check_runtime_env import _bool_literal_check, _settings_check


def _write_env(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_runtime_env_preflight_catches_bool_typo(tmp_path):
    env_file = _write_env(
        tmp_path / ".env",
        "ALLOW_SYNTHETIC_DATA=falses\n",
    )
    errors = _bool_literal_check(env_file)
    assert errors
    assert any("ALLOW_SYNTHETIC_DATA" in err for err in errors)


def test_runtime_env_preflight_accepts_valid_bool(tmp_path):
    env_file = _write_env(
        tmp_path / ".env",
        "ALLOW_SYNTHETIC_DATA=false\n",
    )
    assert _bool_literal_check(env_file) == []
    assert _settings_check(env_file) == []
