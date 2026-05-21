from tests.support.bin import validate_path_solver_munich as validator


def test_reflection_max_bounces_cli_defaults_to_first_order():
    args = validator.build_parser().parse_args([])

    assert args.reflection_max_bounces == 1


def test_reflection_max_bounces_cli_accepts_third_order():
    args = validator.build_parser().parse_args(["--reflection-max-bounces", "3"])

    assert args.reflection_max_bounces == 3
