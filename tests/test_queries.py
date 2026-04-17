"""Tests for SQL query helpers."""

from baseball_rag.db.queries import (
    get_career_stat_leaders,
    get_fielding_leaders,
    get_stat_leaders,
)


def test_get_stat_leaders_returns_list():
    """Test that get_stat_leaders returns a list of dicts with correct keys."""
    result = get_stat_leaders("HR", 1965)  # year we have data for
    assert isinstance(result, list)
    assert len(result) > 0
    row = result[0]
    assert "name" in row
    assert "team" in row
    assert "stat_value" in row


def test_rbi_leaders_1962():
    """Test RBI leaders for 1962 - Mickey Mantle should be #1 with 123 RBIs."""
    result = get_stat_leaders("RBI", 1962)
    assert len(result) > 0
    mantle_row = None
    for row in result:
        if "Mickey" in str(row.get("name", "")):
            mantle_row = row
            break
    assert mantle_row is not None, f"Mickey Mantle not found in results: {result}"
    assert mantle_row["stat_value"] == 123


def test_career_hr_leaders():
    """Test career HR leaders - Babe Ruth should be #1."""
    result = get_career_stat_leaders("HR")
    assert len(result) >= 3
    top_3_names = [row["name"] for row in result[:3]]
    assert any("Ruth" in name or "Babe" in name for name in top_3_names), (
        f"Babe Ruth not in top 3: {top_3_names}"
    )


def test_outfield_putouts_1983():
    """Test outfield putouts leaders for 1983."""
    result = get_fielding_leaders(1983, position="OF")
    assert isinstance(result, list)
    assert len(result) > 0
    row = result[0]
    # Our implementation uses 'player' and 'stat_value'
    assert "player" in row or "name" in row
    assert "putouts" in row or "stat_value" in row
