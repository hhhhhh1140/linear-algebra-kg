"""大模型调用：通过 OpenAI 兼容接口调用云端模型（DeepSeek、通义千问、智谱、Kimi、OpenAI 等都支持）。

配置放在项目根目录的 .llm.env 里（已被 .gitignore 排除，不会提交），格式见 .llm.env.example：
    LLM_BASE_URL=https://api.deepseek.com/v1
    LLM_API_KEY=你的 key
    LLM_MODEL=deepseek-chat
也可以直接用同名环境变量，环境变量优先。
"""
from __future__ import annotations

import json
import http.client
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".llm.env"


def load_config() -> dict:
    cfg = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"').strip("'")
    for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    missing = [k for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL") if not cfg.get(k)]
    if missing:
        raise SystemExit(f"缺少大模型配置 {missing}。请把 .llm.env.example 复制为 .llm.env 并填写。")
    return cfg


class ApiLLM:
    def __init__(self):
        cfg = load_config()
        self.base = cfg["LLM_BASE_URL"].rstrip("/")
        self.key = cfg["LLM_API_KEY"]
        self.name = cfg["LLM_MODEL"]
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        # 先按系统代理设置连接；连不上时自动改为直连（国内接口经海外代理常常握手失败）
        self._openers = [urllib.request.build_opener(),
                         urllib.request.build_opener(urllib.request.ProxyHandler({}))]

    def chat(self, system: str, user: str, max_tokens: int = 256) -> str:
        body = json.dumps({"model": self.name, "temperature": 0, "max_tokens": max_tokens,
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}]}).encode()
        for attempt in range(5):
            # 每次都新建请求：urllib 走代理时会改写 Request 对象，复用它会让直连仍然连到代理
            req = urllib.request.Request(self.base + "/chat/completions", data=body, headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
            try:
                with self._openers[0].open(req, timeout=45) as r:
                    data = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 529) and attempt < 4:   # 限流或服务端错误：退避重试
                    time.sleep(2 ** attempt)
                    continue
                raise SystemExit(f"大模型接口返回 {e.code}: {e.read().decode(errors='replace')[:300]}")
            except (OSError, http.client.HTTPException) as e:   # 连接失败、超时、连接被重置
                if isinstance(e, urllib.error.URLError) and len(self._openers) > 1:      # 换成直连再试，之后一直用直连
                    self._openers.pop(0)
                    continue
                if attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
                raise SystemExit(f"无法连接大模型接口 {self.base}: {e}")
        u = data.get("usage") or {}
        self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        self.usage["completion_tokens"] += u.get("completion_tokens", 0)
        self.usage["calls"] += 1
        self.last_prompt_tokens = u.get("prompt_tokens", len(system) + len(user))
        return (data["choices"][0]["message"].get("content") or "").strip()


def get_llm():
    return ApiLLM()
