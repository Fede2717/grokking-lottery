from scripts.run_parallel_seeds import EXPERIMENT_MODULES, build_command


def test_launcher_selects_matching_hydra_experiment():
    for experiment, module in EXPERIMENT_MODULES.items():
        command = build_command(module, seed=3, extra_args=[], debug=False)
        assert f"experiment={experiment}" in command
        assert "seed=3" in command
