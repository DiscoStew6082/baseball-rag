"""Tests for player_biography routing intent."""

from baseball_rag.routing import route


class TestPlayerBioRouting:
    def test_who_was_wally_pipp(self):
        """'who was Wally Pipp' → player_biography."""
        result = route("who was Wally Pipp")
        assert result.intent == "player_biography"
        assert result.player_name == "Wally Pipp"

    def test_what_teams_did_he_play_for(self):
        """'what teams did he play for' → player_biography (when player context exists)."""
        # This tests that biography questions route correctly
        # Note: without prior context, this might not extract player_name in fallback
        result = route("what teams did he play for")
        assert result.intent == "player_biography"

    def test_tell_me_about_player(self):
        """'tell me about this player' → player_biography."""
        result = route("tell me about this player")
        assert result.intent == "player_biography"

    def test_stat_query_not_biography(self):
        """'how many HRs did Aaron Judge have' → stat_query, not biography."""
        result = route("how many HRs did Aaron Judge have")
        assert result.intent == "stat_query"
        assert result.player_name == "Aaron Judge"
        # Should NOT be player_biography even though a name is present

    def test_rbi_count_not_biography(self):
        """'how many RBIs does Shohei Ohtani have' → stat_query."""
        result = route("how many RBIs does Shohei Ohtani have")
        assert result.intent == "stat_query"
        assert result.player_name == "Shohei Ohtani"

    def test_biography_extracts_player_name(self):
        """player_biography should extract player_name when present."""
        result = route("who was Rogers Hornsby")
        assert result.intent == "player_biography"
        assert result.player_name == "Rogers Hornsby"

    def test_general_explanation_when_no_player_name(self):
        """'what is baseball' → general_explanation, not player_biography."""
        result = route("what is a balk")
        assert result.intent == "general_explanation"
