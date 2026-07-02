#!/bin/sh
# Launch `hexo-a0 serve` from HEXO_* env vars (see the Dockerfile for defaults).
# Optional flags (--url-prefix, --admin-token) are only passed when set, so an
# empty value means "let the server default" (e.g. a random admin token printed
# to stderr at startup) rather than passing an empty flag.
set -eu

if [ ! -f "$HEXO_CHECKPOINT" ]; then
    echo "FATAL: checkpoint not found at $HEXO_CHECKPOINT" >&2
    echo "       mount one, e.g. -v \$PWD/champion.pt:/models/champion.pt:ro,Z" >&2
    exit 1
fi
if [ ! -r "$HEXO_CHECKPOINT" ]; then
    echo "FATAL: checkpoint at $HEXO_CHECKPOINT exists but is not readable by uid $(id -u)." >&2
    echo "       Make it world-readable (chmod a+r), and on SELinux hosts (Fedora/RHEL/" >&2
    echo "       Oracle Linux) add the ',Z' relabel to the mount: ...:/models/champion.pt:ro,Z" >&2
    exit 1
fi

set -- hexo-a0 serve \
    --config "$HEXO_CONFIG" \
    --checkpoint "$HEXO_CHECKPOINT" \
    --db "$HEXO_DB" \
    --port "$HEXO_PORT" \
    --bind "$HEXO_BIND" \
    --mcts-sims "$HEXO_MCTS_SIMS" \
    --model-label "$HEXO_MODEL_LABEL"

[ -n "${HEXO_URL_PREFIX:-}" ] && set -- "$@" --url-prefix "$HEXO_URL_PREFIX"
[ -n "${HEXO_ADMIN_TOKEN:-}" ] && set -- "$@" --admin-token "$HEXO_ADMIN_TOKEN"
# Strength levels: comma-separated sims mapped onto casual,easy,standard,strong
# (e.g. 0,64,128,512), plus which one is the default.
[ -n "${HEXO_DIFFICULTY_SIMS:-}" ] && set -- "$@" --difficulty-sims "$HEXO_DIFFICULTY_SIMS"
[ -n "${HEXO_DEFAULT_DIFFICULTY:-}" ] && set -- "$@" --default-difficulty "$HEXO_DEFAULT_DIFFICULTY"

exec "$@"
