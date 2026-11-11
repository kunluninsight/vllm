PYTHON ?= python3
DIST_DIR ?= dist
# Reuse the local setuptools/wheel; set BUILD_ISOLATION=1 to build in a fresh venv.
BUILD_ISOLATION ?= 0

ifeq ($(BUILD_ISOLATION),0)
BUILD_FLAGS += --no-isolation
endif

.PHONY: all wheel check version clean

all: wheel

wheel: clean
	$(PYTHON) -m build --wheel --outdir $(DIST_DIR) $(BUILD_FLAGS) .
	@ls -l $(DIST_DIR)/*.whl

# Version is derived from the nearest git tag and commit id, see setup.py.
version:
	@$(PYTHON) setup.py --version 2>/dev/null | tail -n1

# hisim.pth must sit at the wheel root so it is installed into site-packages.
check:
	@whl=$$(ls -t $(DIST_DIR)/hisim-*.whl 2>/dev/null | head -n1); \
	test -n "$$whl" || { echo "no wheel in $(DIST_DIR), run 'make wheel' first"; exit 1; }; \
	$(PYTHON) -m zipfile -l "$$whl" | grep -qE '^hisim\.pth ' \
		|| { echo "hisim.pth missing from $$whl root"; exit 1; }; \
	echo "$$whl OK"

clean:
	rm -rf build $(DIST_DIR) src/*.egg-info
