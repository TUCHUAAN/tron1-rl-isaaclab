from isaaclab.terrains import (
    HfInvertedPyramidSlopedTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfWaveTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    MeshRandomGridTerrainCfg,
    TerrainGeneratorCfg,
)

#############################
# Rough Terrain Configuration
#############################

BLIND_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=16,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.25),
        "waves": HfWaveTerrainCfg(proportion=0.25, amplitude_range=(0.01, 0.06), num_waves=10, border_width=0.25),
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.25, grid_width=0.15, grid_height_range=(0.01, 0.04), platform_width=2.0
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.25, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
    },
    curriculum=True,
    difficulty_range=(0.0, 1.0),
)

BLIND_ROUGH_TERRAINS_PLAY_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "waves": HfWaveTerrainCfg(proportion=0.33, amplitude_range=(0.01, 0.06), num_waves=10, border_width=0.25),
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.33, grid_height_range=(0.01, 0.04), platform_width=2.0
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.34, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
    },
    curriculum=False,
    difficulty_range=(1.0, 1.0),
)


##################################
# Hard Rough Terrain Configuration
##################################

BLIND_HARD_ROUGH_TERRAINS_CFG = BLIND_ROUGH_TERRAINS_CFG.copy()
BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["waves"].num_waves = 8
BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["waves"].amplitude_range = (0.02, 0.10)
BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["boxes"].grid_height_range = (0.02, 0.08)
BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["random_rough"].noise_range = (0.02, 0.10)
BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["random_rough"].noise_step = 0.02

BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG = BLIND_ROUGH_TERRAINS_PLAY_CFG.copy()
BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["waves"].num_waves = 8
BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["waves"].amplitude_range = (0.02, 0.10)
BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["boxes"].grid_height_range = (0.02, 0.08)
BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["random_rough"].noise_range = (0.02, 0.10)
BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["random_rough"].noise_step = 0.02


##############################
# Stairs Terrain Configuration
##############################

STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(16.0, 16.0),
    border_width=20.0,
    num_rows=8,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
    curriculum=True,
    difficulty_range=(0.0, 1.0),
)

STAIRS_TERRAINS_PLAY_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(16.0, 16.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
    curriculum=True,
    difficulty_range=(1.0, 1.0),
)


########################################
# Wheelfoot dual-expert terrain curricula
########################################

WHEEL_HEIGHT_PRETRAIN_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=16,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)},
    curriculum=False,
    difficulty_range=(0.0, 0.0),
)

WHEEL_MODE_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    # More rows make the 0--0.4 slope and obstacle progression less abrupt.
    num_rows=12,
    num_cols=16,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.20),
        # Environments spawn on the center platform and move outwards.  The
        # inverted pyramid therefore represents an actual uphill traversal.
        "slope_up": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "slope_down": HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "waves": HfWaveTerrainCfg(
            proportion=0.10, amplitude_range=(0.01, 0.06), num_waves=10, border_width=0.25
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.10, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "stairs_up": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.005, 0.04),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down": MeshPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.03, 0.12),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
    curriculum=True,
    difficulty_range=(0.0, 1.0),
)

WHEEL_MODE_TERRAINS_PLAY_CFG = WHEEL_MODE_TERRAINS_CFG.replace(
    num_rows=4,
    num_cols=4,
    curriculum=False,
    difficulty_range=(0.6, 0.6),
)


FOOT_ALL_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(12.0, 12.0),
    border_width=20.0,
    num_rows=10,
    num_cols=16,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.10),
        "waves": HfWaveTerrainCfg(
            proportion=0.10, amplitude_range=(0.01, 0.08), num_waves=10, border_width=0.25
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(0.01, 0.08), noise_step=0.01, border_width=0.25
        ),
        "stairs_up": MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "slope_up": HfPyramidSlopedTerrainCfg(
            proportion=0.075, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "slope_down": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.075, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
    curriculum=True,
    difficulty_range=(0.0, 1.0),
)

FOOT_ALL_TERRAINS_PLAY_CFG = FOOT_ALL_TERRAINS_CFG.replace(
    num_rows=4,
    num_cols=4,
    curriculum=False,
    difficulty_range=(0.6, 0.6),
)
