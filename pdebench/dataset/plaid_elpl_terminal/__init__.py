"""PLAID el-pl terminal prediction: mesh -> final displacement field."""

from pdebench.dataset.plaid_elpl_terminal.loader import load_plaid_elpl_terminal_dataset
from pdebench.dataset.plaid_elpl_terminal.manifest import terminal_length

__all__ = ["load_plaid_elpl_terminal_dataset", "terminal_length"]
