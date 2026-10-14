# WEANJ — common tasks. Run from the repository root.
JULIA      ?= julia +1.12
QUARTO     ?= quarto
GPU_MAX_MIB ?= 1024

# Source notebooks that execute code (host pages only embed them).
SOURCES := $(wildcard smoke/enzyme.qmd smoke/reactant.qmd smoke/jax.qmd \
                      examples/*/enzyme.qmd examples/*/reactant.qmd examples/*/jax.qmd)

.PHONY: help setup gpu-check run-gpu render preview clean

help:
	@echo "make setup      install Julia 1.12 env, Python env, write _environment.local"
	@echo "make run-gpu    execute every source notebook on the GPU (refreshes _freeze/ and benchmarks/)"
	@echo "make render     build the site from frozen results (no execution)"
	@echo "make preview    live preview in the browser"

setup:
	juliaup add 1.12 || true
	$(JULIA) --project=julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
	cd python && uv sync
	echo "QUARTO_PYTHON=$(CURDIR)/python/.venv/bin/python" > _environment.local

# Benchmarks are only meaningful on an otherwise idle GPU.
gpu-check:
	@used=$$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); \
	if [ $$used -gt $(GPU_MAX_MIB) ]; then \
	  echo "GPU has $$used MiB in use (> $(GPU_MAX_MIB)). Stop other GPU jobs first:"; \
	  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; exit 1; \
	fi; echo "GPU idle ($$used MiB in use)"

# Rendering a single file ignores `freeze`, so this re-executes each notebook.
# Quarto keeps Julia notebook workers alive after a render, and a Reactant worker keeps
# most of the GPU reserved, so stop them between notebooks and wait for an idle GPU.
run-gpu: gpu-check
	@for f in $(SOURCES); do \
	  echo "== $$f"; $(QUARTO) render $$f || exit 1; \
	  pkill -f "julia.*--startup-file=no" || true; sleep 5; \
	  $(MAKE) --no-print-directory gpu-check || exit 1; \
	done
	$(QUARTO) render

render:
	$(QUARTO) render

preview:
	$(QUARTO) preview

clean:
	rm -rf _site .quarto
