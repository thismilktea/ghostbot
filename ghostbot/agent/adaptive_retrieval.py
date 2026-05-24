import json
from pydantic import BaseModel, Field
from ghostbot.providers.base import LLMProvider
from loguru import logger
import asyncio

class QueryIntent(BaseModel):
    need_search: bool = Field(description="是否需要检索本地代码和历史报错。如果是闲聊、纯写新代码，设为 false。")
    temporal_intent: str = Field(description="时序意图：'high'(昨天/刚才/最近), 'low'(一般排查), 'none'(纯算法/底层概念)")
    expanded_keywords: str = Field(description="提取核心技术词并补充2-3个底层术语或缩写，空格分隔。")

class IntentRouter:
    """前置意图分类器"""
    def __init__(self, provider: LLMProvider, model: str):
        self.provider = provider
        self.model = model

    async def analyze(self, raw_query: str) -> QueryIntent:
        logger.debug(f"👉 正在使用的路由模型是: {self.model}")
        # 短文本或极简命令，直接放行，默认需要搜索，无时序敏感
        if len(raw_query) <= 4:
            return QueryIntent(need_search=True, temporal_intent="low", expanded_keywords=raw_query)

        prompt = f"""分析以下开发者提问，并输出 JSON 格式的路由策略：
        用户提问: {raw_query}

        【严格判定规则】
        1. need_search (布尔值):
           - 只要提问中包含"之前"、"上次"、"历史"、"报错"、"刚才"等涉及上下文的词，必须为 true！
           - 只有在纯写新代码（如"帮我写个脚本"）、纯概念解释（"什么是多线程"）或纯打招呼时，才为 false。
        2. temporal_intent: (high/low/none)
        3. expanded_keywords: 提取核心技术词并补充2-3个底层术语或缩写，空格分隔。"""

        try:
            coro = self.provider.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100  # <--- 限制最大输出长度
            )

            # 🔪 绝杀 2：套上 1.5 秒的绝对超时绞刑架！
            # 就算云端排队，我也只等 1.5 秒，超时直接去搜本地 SQLite！
            response = await asyncio.wait_for(coro, timeout=10)
            raw_text = response.content
            print(f"\n🚨 [意图路由 Raw Response]: {raw_text}\n")
            # 解析大模型返回的 JSON
            content = response.content.replace("```json", "").replace("```", "").strip()
            data = json.loads(content)
            return QueryIntent(**data)
        except Exception as e:
            logger.warning(f"⚠️ 意图路由失败，降级为默认搜索: {e}")
            return QueryIntent(need_search=True, temporal_intent="low", expanded_keywords=raw_query)