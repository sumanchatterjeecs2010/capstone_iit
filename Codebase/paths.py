"""
paths.py
--------
Shared filesystem constants for the Multimodal Medical Assistant.

Reuse
-----
Import ``CODEBASE`` or ``CONVERSATION_PATH`` instead of hard-coding paths::

    from paths import CONVERSATION_PATH
"""

import os

# Directory that contains main.py and the rest of the runtime modules.
CODEBASE = os.path.dirname(os.path.abspath(__file__))

# Default structured output written after each CLI/TUI run.
CONVERSATION_PATH = os.path.join(CODEBASE, "conversation.json")
