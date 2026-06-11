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
    {{python}} -m facetrak.api {{port}}

# Start MCP server for LLM integration (stdio)
mcp-serve:
    {{python}} -m facetrak.mcp_server

# Standalone pan-tilt simulation (no camera needed)
sim:
    {{python}} -c "from facetrak.simulation import demo; demo()"

# Install project + dev deps
setup:
    {{pip}} install -e .

# Install single dependency and update pyproject.toml
install dep="":
    {{pip}} install "{{dep}}"
    {{pip}} freeze | grep -ivE "^(opencv|mediapipe|numpy|pillow)" | xargs -r -I{} echo "# {}"
    @echo "Don't forget to add {{dep}} to pyproject.toml manually"

# Show registered faces
list-faces:
    @ls -1 faces/ 2>/dev/null || echo "No faces registered yet"

# Delete a registered person (usage: just forget NAME)
forget name:
    rm -rf faces/"{{name}}"
    {{python}} -c "print('Deleted', '{{name}}')"

# Delete ALL registered faces
clear-faces:
    rm -rf faces/*/
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
    uv sync --upgrade --all-extras

# Show help
default:
    @just --list
