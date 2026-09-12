"""
Small in-memory state shared between server.py (the FastAPI app,
loaded as __main__ when run via `python src/server.py`) and agent.py
(the Agora agent lifecycle wrapper).

Exists as its own module specifically so agent.py can import it directly
at the top of the file: server.py running as __main__ means `from server
import ...` inside agent.py does NOT reach the actual running module --
Python resolves "server" as a fresh, second import of src/server.py
under a different module name, re-running its top-level code (a second
Agent() instance, a second empty channel_names dict) with no error to
show for it. Confirmed live (2026-09-12): a delegate-avatar display name
registered this way was silently written into that ghost copy and never
seen by the real /getNames endpoint. A separate module neither file
runs as __main__ has no such ambiguity.
"""
from typing import Dict

# uid -> display name, per channel. Written by server.py's /setName and
# by agent.py's delegate-avatar startup; read by server.py's /getNames
# (polled by the web client, see ConversationComponent.tsx).
channel_names: Dict[str, Dict[str, str]] = {}
