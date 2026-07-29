# Copyright Xingyu Chen.
# Tests compact Python function signature formatting.

"""Tests compact Python function signature formatting."""

from tools.compact_signatures import format_text


def test_short_signature_collapses_to_one_line() -> None:
    source = "def add(\n    left: int,\n    right: int,\n) -> int:\n    return left + right\n"

    assert format_text(source) == (
        "def add(left: int, right: int) -> int:\n"
        "    return left + right\n"
    )


def test_long_signature_packs_multiple_parameters_per_line() -> None:
    source = (
        "def evaluate(\n"
        "    first_parameter: torch.Tensor,\n"
        "    second_parameter: torch.Tensor,\n"
        "    third_parameter: torch.Tensor,\n"
        "    fourth_parameter: torch.Tensor,\n"
        "    fifth_parameter: torch.Tensor,\n"
        ") -> dict[str, torch.Tensor]:\n"
        "    pass\n"
    )

    formatted = format_text(source)

    assert "first_parameter: torch.Tensor, second_parameter: torch.Tensor," in formatted
    assert max(map(len, formatted.splitlines())) <= 100
    assert format_text(formatted) == formatted


def test_multiline_parameter_expression_is_left_unchanged() -> None:
    source = (
        "def evaluate(\n"
        "    values: tuple[\n"
        "        int,\n"
        "        int,\n"
        "    ],\n"
        ") -> None:\n"
        "    pass\n"
    )

    assert format_text(source) == source
