from pathlib import Path

STAT_DEFS_DIR = Path(__file__).parent / "stat_definitions"
HOF_DIR = Path(__file__).parent / "hof"


def get_stat_defs() -> list[Path]:
    return list(STAT_DEFS_DIR.glob("*.md"))


def get_hof_bios() -> list[Path]:
    return list(HOF_DIR.glob("*.md"))
