"""Tests for sprt_daemon opening-generator selection (noise-off SPRT gate)."""

from hexo_a0.sprt_daemon import _opening_generator_for_pair


T, C = ("trainee_model", "tmc"), ("champion_model", "cmc")


class TestOpeningGeneratorForPair:
    def test_champion_fixed(self):
        assert _opening_generator_for_pair("champion", 0, T, C) == C
        assert _opening_generator_for_pair("champion", 1, T, C) == C

    def test_trainee_fixed(self):
        assert _opening_generator_for_pair("trainee", 0, T, C) == T
        assert _opening_generator_for_pair("trainee", 7, T, C) == T

    def test_alternate_even_trainee_odd_champion(self):
        assert _opening_generator_for_pair("alternate", 0, T, C) == T
        assert _opening_generator_for_pair("alternate", 1, T, C) == C
        assert _opening_generator_for_pair("alternate", 2, T, C) == T
        assert _opening_generator_for_pair("alternate", 3, T, C) == C
