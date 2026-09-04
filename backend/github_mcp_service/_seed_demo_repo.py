import os
import asyncio
import base64
import json

import httpx2
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

token = os.getenv("GITHUB_TOKEN")
REPO = "HARDIK-TSH1392/stark-payment-global"
HEADERS = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

commits = [
    ("README.md", "# Stark Payment Global\n\nCore payment gateway and checkout processing service.\n",
     "chore: initial project scaffold"),
    ("src/payment_gateway.py",
     "def process_payment(order_id, amount, currency):\n    \"\"\"Routes a payment through the configured provider.\"\"\"\n    raise NotImplementedError\n",
     "feat: add payment_gateway module skeleton"),
    ("src/db_config.py",
     "DB_POOL_CONFIG = {\n    \"max_connections\": 50,\n    \"timeout_seconds\": 5,\n}\n",
     "fix: increase payment DB connection pool timeout from 2s to 5s"),
    ("requirements.txt",
     "payment-sdk==4.2.1\nrequests==2.31.0\n",
     "chore: bump payment-sdk to 4.2.1"),
    ("src/webhook_handler.py",
     "def handle_bank_webhook(payload):\n    if payload is None:\n        return {\"status\": \"ignored\"}\n    return {\"status\": \"processed\"}\n",
     "fix: handle null response from bank webhook callback"),
]


async def main():
    async with httpx2.AsyncClient() as client:
        for path, content, message in commits:
            b64 = base64.b64encode(content.encode()).decode()
            resp = await client.put(
                f"https://api.github.com/repos/{REPO}/contents/{path}",
                headers=HEADERS,
                json={"message": message, "content": b64, "branch": "main"},
            )
            print(path, "->", resp.status_code, message)
            if resp.status_code not in (200, 201):
                print(json.dumps(resp.json(), indent=2)[:500])
            await asyncio.sleep(1)


asyncio.run(main())
