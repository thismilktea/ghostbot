# 文件路径: ghostbot/channels/ghost.py

import os
import re
import time
import asyncio
import threading
from loguru import logger
from ghostbot.channels.base import BaseChannel
from ghostbot.bus.events import OutboundMessage
from ghostbot.agent.error_cache import should_diagnose
from ghostbot.utils.toast import win_toast


class GhostChannel(BaseChannel):
    name = "ghost"
    display_name = "Ghost IDE Watcher"

    def __init__(self, config, bus):
        super().__init__(config, bus)
        self.ide_log_path = config.get("log_path", r"C:\temp\ghost_ide_run.log")

    def is_allowed(self, sender_id: str) -> bool:
        return True

    async def start(self) -> None:
        self._running = True
        logger.info("📡 Ghost 通道已启动，双重雷达开启！")
        self._main_loop = asyncio.get_running_loop()

        self.watch_targets = [
            r"C:\temp\ghost_ide_run.log",
            r"C:\temp\ghost_powershell.log"
        ]
        for log_file in self.watch_targets:
            threading.Thread(target=self._watchdog_loop, args=(log_file,), daemon=True).start()

    async def stop(self) -> None:
        self._running = False
        logger.info("📡 Ghost 雷达已关闭。")

    def _watchdog_loop(self, filepath: str):
        """独立的雷达保安（带多行防抖收集功能 + 防弹衣）"""
        if not os.path.exists(filepath):
            open(filepath, 'a').close()

        file_encoding = 'gbk' if 'powershell' in filepath.lower() else 'utf-8'

        try:
            with open(filepath, 'r', encoding=file_encoding, errors='replace') as f:
                f.seek(0, 2)  # 初始启动时跳到末尾
                buffer = []

                is_collecting_error = False
                silence_ticks = 0

                logger.info(f"👀 线程已就绪，正在死盯文件: {filepath}")

                while self._running:
                    # 💡 修复点：这里必须包裹一个总的 try，与底部的 except Exception as inner_e 呼应
                    try:
                        # ==========================================
                        # 🚀 新增：防截断/覆盖检测逻辑 (Truncation Detection)
                        # ==========================================
                        try:
                            current_position = f.tell()
                            file_size = os.path.getsize(filepath)

                            # 如果当前指针比文件本身还大，说明文件被 IDE 'w' 模式清空并重写了！
                            if current_position > file_size:
                                logger.debug(f"文件被清空，重置指针: {filepath}")
                                f.seek(0, 0)  # 指针立刻回到文件开头！
                                buffer.clear()  # 清空旧的脏数据
                                is_collecting_error = False
                                silence_ticks = 0
                        except OSError:
                            # 防止文件刚好被删掉的那一瞬间报错
                            time.sleep(0.1)
                            continue
                        # ==========================================

                        line = f.readline()

                        # 【情况 A：没有新日志写入】
                        if not line:
                            if is_collecting_error:
                                silence_ticks += 1
                                # 发现报错后，超过 1秒（2次*0.5s）没有新日志，说明报错打印完了！
                                if silence_ticks >= 2:
                                    logger.info("👻 雷达收集完毕，准备验证并发送给大脑...")
                                    error_context = "".join(buffer[-50:])

                                    if should_diagnose(error_context):
                                        logger.info("✅ 缓存验证通过！正在跨线程投递消息...")
                                        asyncio.run_coroutine_threadsafe(
                                            self._handle_message(
                                                sender_id="local_dev",
                                                chat_id="desktop_toast",
                                                content=error_context[-1500:],
                                                metadata={"is_ghost_mode": True}
                                            ),
                                            self._main_loop
                                        )
                                    else:
                                        # 🚨 重点：如果被缓存拦截，这里会大声告诉你！
                                        logger.warning(
                                            "🚫 该报错被 should_diagnose 缓存拦截！（你是不是重复触发了同一个报错？）")

                                    # 状态重置
                                    buffer.clear()
                                    is_collecting_error = False
                                    silence_ticks = 0

                            time.sleep(0.5)
                            continue

                        # 【情况 B：有新日志写入】
                        buffer.append(line)

                        # 🚨 修复潜在的内存泄漏：让 buffer 永远保持在最近 100 行以内
                        if len(buffer) > 100:
                            buffer.pop(0)

                        # 如果匹配到了报错关键词，进入收集模式
                        if re.search(r'(Exception|Error|Traceback|BUILD FAILURE)', line, re.I):
                            if not is_collecting_error:
                                logger.info(f"🚨 雷达捕获到异常关键词，开始防抖收集... ({filepath})")
                            is_collecting_error = True
                            silence_ticks = 0  # 重置静默倒计时

                    # 💡 修复点：内部的异常捕获与 while 循环内部的逻辑对齐
                    except Exception as inner_e:
                        logger.error(f"⚠️ 雷达循环内部发生错误: {inner_e}")
                        time.sleep(1)  # 睡一秒防止死循环刷屏

        except Exception as e:
            logger.error(f"💀 致命错误：Ghost 雷达线程彻底崩溃: {e}")

    async def send(self, msg: OutboundMessage) -> None:
        """大脑走完标准的推理流程后，消息会流回这里进行弹窗"""
        win_toast(msg.content)