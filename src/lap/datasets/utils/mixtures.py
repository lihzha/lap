OXE_NAMED_MIXTURES: dict[str, list[tuple[str, float]]] = {
    "oxe_magic_soup": [
        ("bc_z", 0.05),
        ("droid", 2.0),
        ("fractal20220817_data", 1.0),
        ("bridge_v2_oxe", 1.0),
        ("taco_play", 2.0),
        (
            "jaco_play",
            1.0,
        ),  # gripper state and action still seems incorrect. Action sometimes should be 1 but is 0. State seems random. Ignore for now.
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.05),
        ("utaustin_mutex", 1.0),
        ("berkeley_fanuc_manipulation", 2.0),  # not sure quaternion is xyzw or wxyz
        ("fmb", 0.05),
        ("berkeley_autolab_ur5", 1.0),
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0),  # no language instructions, 50 trajs
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),  # no language instructions, 250 trajs
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),  # no language instructions, 600 trajs
        ("viola", 1.0),  # gripper mostly out of view, 135 trajs
        ("molmoact_dataset", 1.0),
    ],
    "libero_finetune": [
        ("libero_10_no_noops", 1.0),
        ("libero_spatial_no_noops", 1.0),
        ("libero_object_no_noops", 1.0),
        ("libero_goal_no_noops", 1.0),
    ],
    # === Individual Datasets (for isolated visualization/testing) ===
    "bc_z": [("bc_z", 1.0)],
    "fractal20220817_data": [("fractal20220817_data", 1.0)],
    "bridge_v2_oxe": [("bridge_v2_oxe", 1.0)],
    "taco_play": [("taco_play", 1.0)],
    "jaco_play": [("jaco_play", 1.0)],
    "furniture_bench_dataset_converted_externally_to_rlds": [
        ("furniture_bench_dataset_converted_externally_to_rlds", 1.0)
    ],
    "utaustin_mutex": [("utaustin_mutex", 1.0)],
    "berkeley_fanuc_manipulation": [("berkeley_fanuc_manipulation", 1.0)],
    "cmu_stretch": [("cmu_stretch", 1.0)],
    "fmb": [("fmb", 1.0)],
    "dobbe": [("dobbe", 1.0)],
    "berkeley_autolab_ur5": [("berkeley_autolab_ur5", 1.0)],
    "bridge": [
        ("bridge_v2_oxe", 1.0),  # Version of Bridge V2 in Open-X GCP Bucket
        # ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
    ],
    "droid": [
        ("droid", 1.0),
    ],
}
