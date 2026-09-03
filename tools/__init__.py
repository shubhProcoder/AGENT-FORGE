"""Tools package — MCP-style tool implementations.

Each sub-package (customer/, order/, document/) contains tool functions
decorated with @register_tool. Importing this package auto-discovers
and registers all tools.
"""

# Auto-import all tool modules so they self-register
from tools import customer, document, order  # noqa: F401
