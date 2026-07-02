"""Serving module: the `hexo-a0 serve` public play + analysis server.

A clean, factored re-implementation of the play-server internals (ported from
the frozen scripts/play_server.py) plus the eval/analysis core (mirroring the
committed policy_viewer spike). The frozen scripts remain as fallbacks; see
docs/superpowers/plans/2026-06-16-public-analysis-tool.md.
"""
