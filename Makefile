.PHONY: install clean

PREFIX ?= $(HOME)/.qntx
BUILD_DIR = build
# A QNTX source tree: CMake compiles plugin/grpc/protocol from it (see README, Build).
QNTX_SRC ?= $(error QNTX_SRC is not set: make install QNTX_SRC=<QNTX checkout>)

# Restart — tell running QNTX to hot-restart gaze. The node's port is BACKEND_PORT, else 8770.
define restart-plugin
	@PORT=$${BACKEND_PORT:-8770}; \
	 curl -sf -X POST http://127.0.0.1:$$PORT/api/plugins/gaze/restart > /dev/null 2>&1 \
		&& echo "  ↻ LOADED — gaze restarted, live at http://127.0.0.1:$$PORT" \
		|| echo "  ⊘ NOT LOADED — no QNTX answered at http://127.0.0.1:$$PORT; the binary is installed, not running"
endef

install: ## Build, install, and restart gaze plugin
	@echo "Building qntx-gaze-plugin..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Release -DQNTX_SRC=$(QNTX_SRC)
	@cd $(BUILD_DIR) && cmake --build . --parallel 5
	@mkdir -p $(PREFIX)/plugins
	@cp $(BUILD_DIR)/qntx-gaze-plugin $(PREFIX)/plugins/qntx-gaze-plugin
	@chmod +x $(PREFIX)/plugins/qntx-gaze-plugin
	@echo "✓ BUILD OK — binary installed to $(PREFIX)/plugins/qntx-gaze-plugin"
	$(call restart-plugin)

clean: ## Clean build artifacts
	@rm -rf $(BUILD_DIR)
	@echo "✓ Clean"
