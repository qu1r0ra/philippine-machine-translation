.PHONY: clean-checkpoints clean-all

# Remove all .ipynb_checkpoints EXCEPT inside notebooks/
clean-checkpoints:
	@echo "Removing .ipynb_checkpoints outside notebooks/..."
	@find . -type d -name ".ipynb_checkpoints" ! -path "./notebooks/*" -exec rm -rf {} +
	@echo "Done. Checkpoints outside notebooks/ removed."

# Remove ALL .ipynb_checkpoints (including inside notebooks/)
clean-all:
	@echo "Removing ALL .ipynb_checkpoints (including notebooks/)..."
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	@echo "Done. All checkpoints removed."
