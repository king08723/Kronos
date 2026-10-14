"""
backtest/core/logger.py
量化回测与实盘统一日志系统
支持：
- 控制台分级彩色诊断
- 全局日志落盘 (kronos_backtest.log)
- 单标的独立日志动态挂载与安全卸载 ({stock_code}_backtest.log)
- 历史日志安全清空与静默模式
"""

from __future__ import annotations

import os
import sys
import glob
import logging
from typing import Optional, List

from .market_rules import SymbolResolver

BACKTEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOG_DIR = os.path.join(BACKTEST_DIR, "output")


def setup_logger(
    log_dir: str = None,
    stock_code: str = None,
    enabled: bool = True,
    file_logging: bool = True,
) -> logging.Logger:
    """
    配置支持控制台与文件落盘的全局诊断 Logger。
    - 控制台输出: INFO 及以上级别 (当 enabled=True 时输出)
    - 全局日志文件 kronos_backtest.log: DEBUG 级别全量记录 (当 file_logging=True 时记录)
    - 单标的日志文件 {stock_code}_backtest.log: 若指定 stock_code 且 file_logging=True，同时独立落盘
    """
    logger = logging.getLogger("KronosBacktest")
    logger.setLevel(logging.DEBUG if enabled else (logging.CRITICAL + 10))

    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    # 控制台 StreamHandler (确保仅添加一次)
    if not any(type(h) is logging.StreamHandler for h in logger.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    global_log_file = os.path.abspath(os.path.join(log_dir, "kronos_backtest.log"))

    if file_logging and enabled:
        # 全局汇总日志 kronos_backtest.log：若切换了目录，关闭并卸载旧汇总 Handler
        for h in list(logger.handlers):
            if isinstance(h, logging.FileHandler):
                bf = getattr(h, 'baseFilename', None)
                if bf and os.path.basename(bf) == "kronos_backtest.log" and bf != global_log_file:
                    try:
                        h.flush()
                        h.close()
                    except Exception:
                        pass
                    logger.removeHandler(h)

        # 全局汇总日志 kronos_backtest.log (确保仅添加一次)
        if not any(getattr(h, 'baseFilename', None) == global_log_file for h in logger.handlers):
            fh = logging.FileHandler(global_log_file, encoding='utf-8', mode='a')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        if stock_code:
            attach_stock_logger(logger, stock_code, log_dir, enabled=True)
    else:
        # 如果关闭文件落盘，安全移除已存在的 FileHandler
        for h in list(logger.handlers):
            if isinstance(h, logging.FileHandler):
                try:
                    h.flush()
                    h.close()
                except Exception:
                    pass
                logger.removeHandler(h)

    return logger


def attach_stock_logger(
    logger: logging.Logger,
    stock_code: str,
    log_dir: str = None,
    enabled: bool = True,
) -> Optional[logging.Handler]:
    """为指定标的附加独立的日志 Handler，输出至 {log_dir}/{stock_code}_backtest.log"""
    if not enabled:
        return None
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    clean_code, _, _ = SymbolResolver.resolve(stock_code)
    clean_code = clean_code or stock_code
    stock_log_file = os.path.abspath(os.path.join(log_dir, f"{clean_code}_backtest.log"))

    for h in logger.handlers:
        if getattr(h, 'baseFilename', None) == stock_log_file:
            return h

    # 挂载前先安全清理同名标的在其他路径的旧句柄（防止跨路径重复记录）
    detach_stock_logger(logger, stock_code=clean_code)

    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    sfh = logging.FileHandler(stock_log_file, encoding='utf-8', mode='a')
    sfh.setLevel(logging.DEBUG)
    sfh.setFormatter(formatter)
    logger.addHandler(sfh)
    return sfh


def detach_stock_logger(
    logger: logging.Logger = None,
    stock_code: str = None,
    log_dir: str = None,
) -> None:
    """安全移除并关闭指定标的（或全部标的）的独立日志 FileHandler"""
    if logger is None:
        logger = logging.getLogger("KronosBacktest")

    clean_code = None
    target_filename = None
    if stock_code:
        clean_code, _, _ = SymbolResolver.resolve(stock_code)
        clean_code = clean_code or stock_code
        target_filename = f"{clean_code}_backtest.log"

    target_file = None
    if log_dir and clean_code:
        target_file = os.path.abspath(os.path.join(log_dir, target_filename))

    handlers_to_remove = []
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            bf = getattr(h, 'baseFilename', None)
            if not bf:
                continue
            # 严格保留全局/批次主汇总日志 (kronos_backtest.log)
            if os.path.basename(bf) == "kronos_backtest.log":
                continue

            if stock_code:
                # 匹配完整路径或文件名（避免因 log_dir 绝对路径差异遗漏）
                if (target_file and bf == target_file) or (os.path.basename(bf) == target_filename):
                    handlers_to_remove.append(h)
            else:
                # 未指定标的，则清理所有标的的独立日志 Handler
                handlers_to_remove.append(h)

    for h in handlers_to_remove:
        try:
            h.flush()
            h.close()
        except Exception:
            pass
        logger.removeHandler(h)


def detach_all_stock_loggers(logger: logging.Logger = None) -> None:
    """便利函数：安全移除并关闭全局 Logger 上所有单标的独立 Handler"""
    detach_stock_logger(logger=logger, stock_code=None)


def set_logging_state(
    enabled: bool = True,
    file_logging: bool = True,
    logger: logging.Logger = None,
    log_dir: str = None,
) -> None:
    """
    统一设置日志开关状态：
    - enabled=False: 全局静默（终端和文件均不输出）
    - enabled=True, file_logging=False: 控制台输出，但关闭文件落盘
    - enabled=True, file_logging=True: 控制台输出且开启文件落盘
    """
    if logger is None:
        logger = logging.getLogger("KronosBacktest")
    if not enabled:
        logger.setLevel(logging.CRITICAL + 10)
        return

    logger.setLevel(logging.DEBUG)
    if not file_logging:
        for h in list(logger.handlers):
            if isinstance(h, logging.FileHandler):
                try:
                    h.flush()
                    h.close()
                except Exception:
                    pass
                logger.removeHandler(h)
    else:
        if log_dir is None:
            log_dir = DEFAULT_LOG_DIR
        global_log_file = os.path.abspath(os.path.join(log_dir, "kronos_backtest.log"))
        if not any(getattr(h, 'baseFilename', None) == global_log_file for h in logger.handlers):
            formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            fh = logging.FileHandler(global_log_file, encoding='utf-8', mode='a')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)


def clear_logs(log_dir: str = None, stock_code: str = None, clear_all: bool = False) -> List[str]:
    """
    清空回测日志文件内容（安全截断，不破坏文件句柄）：
    - 若 clear_all=True: 清空 log_dir 下全部 *.log 文件（含 kronos_backtest.log 和所有标的日志）
    - 若指定 stock_code: 仅清空对应 {stock_code}_backtest.log
    - 若 stock_code=None 且 clear_all=False: 清空 kronos_backtest.log

    返回被清空的文件绝对路径列表。
    """
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    # 1. 刷新所有 logger handler
    main_logger = logging.getLogger("KronosBacktest")
    for h in main_logger.handlers:
        try:
            h.flush()
        except Exception:
            pass

    cleared = []
    if clear_all:
        pattern = os.path.join(log_dir, "*.log")
        for fpath in glob.glob(pattern):
            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.truncate(0)
                cleared.append(os.path.abspath(fpath))
            except Exception as e:
                main_logger.warning(f"清空日志文件失败: {fpath}, 原因: {e}")
    elif stock_code:
        clean_code, _, _ = SymbolResolver.resolve(stock_code)
        clean_code = clean_code or stock_code
        fpath = os.path.join(log_dir, f"{clean_code}_backtest.log")
        if os.path.exists(fpath):
            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.truncate(0)
                cleared.append(os.path.abspath(fpath))
            except Exception as e:
                main_logger.warning(f"清空标的日志文件失败: {fpath}, 原因: {e}")
    else:
        fpath = os.path.join(log_dir, "kronos_backtest.log")
        if os.path.exists(fpath):
            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.truncate(0)
                cleared.append(os.path.abspath(fpath))
            except Exception as e:
                main_logger.warning(f"清空全局日志文件失败: {fpath}, 原因: {e}")

    return cleared
