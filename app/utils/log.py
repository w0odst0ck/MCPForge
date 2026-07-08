"""日志工具 — 封装 loguru，支持文件轮转和控制台输出。

用法:
    from app.utils.log import log
    log.info("服务启动")
    log.error("连接失败", exc_info=True)
"""
import sys

from loguru import logger as _logger

from app.config import settings


class Loggers:
    """日志初始化器，在应用启动时调用。"""

    _initialized = False

    @classmethod
    def init_config(cls) -> None:
        if cls._initialized:
            return

        # 移除默认 handler
        _logger.remove()

        # 控制台输出
        _logger.add(
            sys.stdout,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
            level=settings.LOG_LEVEL,
            colorize=True,
        )

        # 文件输出（每日轮转，保留 30 天）
        _logger.add(
            settings.LOG_FILE,
            rotation="1 day",
            retention="30 days",
            compression="gz",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                "{name}:{function}:{line} - {message}"
            ),
            level=settings.LOG_LEVEL,
        )

        cls._initialized = True


# 模块级引用，方便直接 import log
log = _logger
