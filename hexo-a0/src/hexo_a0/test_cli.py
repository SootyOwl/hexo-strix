"""Tests for the CLI entry point."""


def test_default_config_command():
    from hexo_a0.cli import main

    # Should print TOML to stdout and return 0
    result = main(["default-config"])
    assert result == 0


def test_train_missing_config():
    from hexo_a0.cli import main

    result = main(["train", "--config", "/nonexistent/path.toml"])
    assert result == 1
