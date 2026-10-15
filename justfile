# ──────────────────────────────────────────────
# FaceTrak — just command runner
# ──────────────────────────────────────────────
venv := ".venv"
python := venv / "bin" / "python"
pip := venv / "bin" / "pip"

# Run the tracker app
run:
    {{python}} -m facetrak

# Start HTTP REST API server (port 8765 by default)
api-serve port="8765":
    {{python}} -m facetrak.servers.rest {{port}}

# Start MCP server for LLM integration (stdio)
mcp-serve:
    {{python}} -m facetrak.servers.mcp

# Standalone pan-tilt simulation (no camera needed)
sim:
    {{python}} -c "from facetrak.viz.simulation import demo; demo()"

# Install project + dev deps
setup:
    {{pip}} install -e .

# Run full test suite
test:
    {{python}} -m pytest tests/ -q

# Lint changed/new Python code (repo-wide baseline has pre-existing findings)
lint:
    ruff check facetrak tests scripts

# Auto-fix import order in touched modules only
lint-fix:
    ruff check --fix --select I,E9,F facetrak tests scripts

# Reproducible performance benchmark (no camera needed)
bench frames="200":
    {{python}} scripts/benchmark.py --frames {{frames}}

# Purge rows older than retention_days via API (needs server + token)
purge token="":
    curl -s -X POST -H "Authorization: Bearer {{token}}" http://127.0.0.1:8765/retention/purge; echo

# Install single dependency and update pyproject.toml
install dep="":
    {{pip}} install "{{dep}}"
    {{pip}} freeze | grep -ivE "^(opencv|mediapipe|numpy|pillow)" | grep . | xargs -I{} echo "# {}"
    @echo "Don't forget to add {{dep}} to pyproject.toml manually"

# Show registered faces
list-faces:
    @ls -1 faces/data/ 2>/dev/null || echo "No faces registered yet"

# Delete a registered person (usage: just forget NAME)
forget name:
    #!/usr/bin/env bash
    set -u
    if [ -z "{{name}}" ]; then echo "usage: just forget NAME"; exit 1; fi
    safe="$(printf '%s' '{{name}}' | tr ' ' '_' | tr -cd 'A-Za-z0-9_-')"
    if [ -z "$safe" ]; then echo "invalid name"; exit 1; fi
    rm -f "faces/data/${safe}.npy"
    echo "Deleted {{name}} (if it existed)"

# Delete ALL registered faces
clear-faces:
    rm -f faces/data/*.npy
    @echo "All faces deleted"

# Update dependency versions in pyproject.toml
sync:
    {{pip}} install -e .
    {{pip}} list --format=freeze | grep -E "^(opencv-contrib-python|mediapipe|numpy|pillow)" > /tmp/_deps.txt
    while IFS= read -r dep; do \
        name=$${dep%%==*}; version=$${dep##*==}; \
        sed -i '' "s/$name>=[0-9.]*/$name>=$version/" pyproject.toml; \
    done < /tmp/_deps.txt
    @echo "pyproject.toml versions updated"

# Clean caches, temp files and remove .egg-link
clean:
    rm -rf __pycache__ .mypy_cache *.egg-info
    rm -f snapshot_*.png
    @echo "Cleaned"

update:
    uv pip install -e "." --upgrade

# Show help
default:
    @just --list
