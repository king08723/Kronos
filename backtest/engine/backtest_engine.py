"""
backtest/engine/backtest_engine.py
量化回测核心执行引擎 (Backtest Engine & KronosBacktesterPro)
具备：
- 完整多市场画像切换 (A股/港股/美股) 与微观摩擦建模
- 多周期支持 (日线/15m/30m) 与自适应时间折算
- 策略插件化架构 (支持接入 Kronos AI、经典技术指标、多因子策略)
- 离散状态机撮合、ATR风险预算与多重出场防守
- 产物导出与全维度量化指标归因
"""

from __future__ import annotations

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from ..core.types import TimeFrame, Position
from ..core.market_rules import SymbolResolver, MarketProfile
from ..core.logger import attach_stock_logger, detach_stock_logger, clear_logs
from ..execution.sizer import RiskBudgetSizer
from ..execution.risk_manager import ExitStateMachine
from ..execution.simulated_broker import SimulatedBroker
from ..data.data_fetcher import DataFetcher
from ..data.kronos_predictor import KronosDataFetcher
from ..strategies.kronos_strategy import KronosStrategy, compute_dynamic_rolling_quantile
from ..engine.metrics import MetricsEvaluator
from ..engine.skill_evaluator import SkillEvaluator
from ..visualizer.plot_static import plot_backtest_results
from ..visualizer.plot_interactive import plot_backtest_interactive, HAS_PLOTLY

logger = logging.getLogger("KronosBacktest")

class KronosBacktesterPro:
    """
    Kronos 量化回测引擎 Pro 版

    现代自适应量化架构：
    - 多市场微观画像 (A股/港股/美股)
    - 样本外时序防偷价 (t-1收盘决策 -> t日开盘撮合)
    - 滚动分位数自校准信号强度
    - 动态 ATR 止损 + 预测波峰主动止盈
    - 一手保底机制 (Min-Lot Guarantee)
    """

    def __init__(
        self,
        data_dir: str,
        model_dir: str,
        initial_capital: float = None,
        commission_rate: float = None,
        min_commission: float = None,
        stamp_duty_rate: float = None,
        slippage_rate: float = None,
        market_profile: 'MarketProfile' = None,
        enable_log: bool = True,
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir

        # 市场规则画像：默认 A 股，可由 set_market 依股票代码自动切换
        self.profile = market_profile or MarketProfile.for_market('CN')
        self.initial_capital = initial_capital if initial_capital is not None else self.profile.default_capital
        self.current_stock_code = None
        self._stock_log_handler = None
        self.log_dir = None
        self.enable_log = enable_log
        # 滑点：未显式指定时跟随市场默认（A 0.10% / HK 0.15% / US 0.05%）
        self._slippage_override = slippage_rate is not None
        self.slippage_rate = (
            slippage_rate if self._slippage_override else self.profile.default_slippage
        )

        # 显式传入的费率覆盖 profile 默认值
        self._override = {}
        if commission_rate is not None:
            self.profile.commission_rate = commission_rate
            self._override['commission_rate'] = commission_rate
        if min_commission is not None:
            self.profile.min_commission = min_commission
            self._override['min_commission'] = min_commission
        if stamp_duty_rate is not None:
            self.profile.stamp_duty_sell = stamp_duty_rate
            self._override['stamp_duty_sell'] = stamp_duty_rate

    def cleanup_logger(self):
        """安全释放当前标的的独立日志 Handler，并确保全局 Logger 不残留此标的的句柄"""
        if self._stock_log_handler:
            try:
                self._stock_log_handler.flush()
                self._stock_log_handler.close()
                logger.removeHandler(self._stock_log_handler)
            except Exception:
                pass
            self._stock_log_handler = None

        if self.current_stock_code:
            try:
                detach_stock_logger(logger, stock_code=self.current_stock_code, log_dir=self.log_dir)
            except Exception:
                pass

    def set_market(
        self,
        stock_code: str,
        market: str = None,
        stock_name: str = None,
        log_dir: str = None,
        enable_log: bool = None,
    ):
        """依股票代码/名称切换市场规则，并保留用户显式覆盖的费率项"""
        if enable_log is not None:
            self.enable_log = enable_log
        if log_dir is not None:
            self.log_dir = log_dir

        # 切换标的前，先彻底释放前一个标的的独立 Handler
        self.cleanup_logger()

        clean_code, _, _ = SymbolResolver.resolve(stock_code)
        self.current_stock_code = clean_code or stock_code
        self.profile = MarketProfile.for_stock(
            stock_code, market=market, stock_name=stock_name
        )
        for k, v in self._override.items():
            setattr(self.profile, k, v)
        # 未手动指定滑点时，跟随新市场默认值
        if not self._slippage_override:
            self.slippage_rate = self.profile.default_slippage

        # 根据开关挂载单标的独立日志 Handler
        if self.enable_log:
            try:
                self._stock_log_handler = attach_stock_logger(
                    logger, self.current_stock_code, self.log_dir, enabled=True
                )
            except Exception:
                pass

        logger.info(f"[{self.current_stock_code}] [市场规则] {stock_code} -> {self.profile.describe()}")
        return self.profile

    @property
    def commission_rate(self):
        return self.profile.commission_rate

    @property
    def min_commission(self):
        return self.profile.min_commission

    @property
    def stamp_duty_rate(self):
        return self.profile.stamp_duty_sell

    # ------------------------------------------------------------------
    # 1. 数据加载
    # ------------------------------------------------------------------

    def load_data(self, stock_code: str, particle_filter: bool = False) -> tuple:
        """加载并标准化历史行情与模型预测数据"""
        clean_code, det_market, _ = SymbolResolver.resolve(stock_code, market=self.profile.market)
        if self.profile.market != det_market:
            self.set_market(clean_code, market=det_market, log_dir=self.log_dir, enable_log=self.enable_log)
        hist_df, _ = DataFetcher.load_stock_history(stock_code, data_dir=self.data_dir, market=self.profile.market)
        pred_df = DataFetcher.load_prediction_data(stock_code, model_dir=self.model_dir, market=self.profile.market, particle_filter=particle_filter)
        return hist_df, pred_df

    def align_and_generate_signals(self, hist_df: pd.DataFrame, pred_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """时序对齐与信号生成 (委托至 KronosStrategy)"""
        strategy = KronosStrategy(profile=self.profile)
        return strategy.align_and_generate_signals(hist_df, pred_df, **kwargs)

    @staticmethod
    def evaluate_prediction_skill(df: pd.DataFrame, hard_stop_loss: float = 0.04) -> dict:
        return SkillEvaluator.evaluate(df, hard_stop_loss=hard_stop_loss)

    @staticmethod
    def print_skill_report(skill: dict, stock_code: str = ''):
        SkillEvaluator.print_report(skill, stock_code=stock_code)

    def check_causality(self, hist_df: pd.DataFrame, pred_df: pd.DataFrame, n_probe: int = 12, **kwargs) -> dict:
        strategy = KronosStrategy(profile=self.profile)
        return strategy.check_causality(hist_df, pred_df, n_probe=n_probe, **kwargs)

    def run_validation_protocol(self, hist_df: pd.DataFrame, pred_df: pd.DataFrame, stock_code: str = '', **base_kwargs) -> dict:
        strategy = KronosStrategy(profile=self.profile)
        return strategy.run_validation_protocol(hist_df, pred_df, stock_code=stock_code, **base_kwargs)

    def evaluate_metrics(self, backtest_df: pd.DataFrame, trades: list) -> dict:
        """计算全套量化绩效指标"""
        return MetricsEvaluator.evaluate(
            backtest_df=backtest_df,
            trades=trades,
            initial_capital=self.initial_capital,
            profile=self.profile,
        )

    def plot_backtest_results(self, signal_df: pd.DataFrame, backtest_df: pd.DataFrame, metrics: dict, trades: list, stock_code: str, output_dir: str, show_plot: bool = False):
        return plot_backtest_results(
            signal_df=signal_df,
            backtest_df=backtest_df,
            metrics=metrics,
            trades=trades,
            stock_code=stock_code,
            output_dir=output_dir,
            profile=self.profile,
            show_plot=show_plot,
        )

    def plot_backtest_interactive(self, signal_df: pd.DataFrame, backtest_df: pd.DataFrame, metrics: dict, trades: list, stock_code: str, z_score_threshold: float = 1.2, crosshair_mode: str = "x"):
        return plot_backtest_interactive(
            signal_df=signal_df,
            backtest_df=backtest_df,
            metrics=metrics,
            trades=trades,
            stock_code=stock_code,
            z_score_threshold=z_score_threshold,
            crosshair_mode=crosshair_mode,
            profile=self.profile,
        )

    def run_backtest_engine(
        self,
        df: pd.DataFrame,
        hard_stop_loss: float = 0.04,
        trailing_stop: float = 0.025,
        trailing_stop_min_profit: float = 0.02,
        max_holding_days: int = 5,
        min_lot_guarantee: bool = True,
        risk_per_trade: float = 0.01,
        max_position_pct: float = 0.80,
        k_stop: float = 1.5,
        k_trail: float = 1.2,
        min_stop: float = 0.02,
        max_stop: float = 0.12,
        intraday_stop: bool = True,
        consecutive_loss_cut: int = 2,
    ) -> tuple:
        """
        逐日仿真撮合引擎（状态机）。

        【撮合规则设计】
        - 开仓/平仓撮合价格：使用当日开盘价（open）乘滑点 (1 ± slippage_rate)。
          因为 exec_signal / pending_close_reason 是前一日收盘后计算出的决策，在今日开盘立即撮合。
        - 每日盯市结算（Mark-to-Market）：以当日真实收盘价 close 结算总账户资产 portfolio_value。
        - 交易约束：多市场制度全真建模（A股100股整手/港股自定义每手/美股1股，T+1/T+0，印花税及交易所规费）。

        参数示例 (Parameters)
        --------------------
        df : pd.DataFrame
            信号对齐矩阵（由 align_and_generate_signals 产出，含 open/close/exec_signal 等列）
        hard_stop_loss : float, default=0.04
            可配置硬止损线（相比成本价）。示例: 0.04 表示亏损达 -4.0% 时止损
        trailing_stop : float, default=0.025
            移动止盈回撤阈值。示例: 0.025 表示自持仓期间最高价回撤 -2.5% 时离场
        trailing_stop_min_profit : float, default=0.02
            移动止盈启动门槛。示例: 0.02 表示必须在浮盈超过 +2.0% 后才激活峰值回撤止盈
        max_holding_days : int, default=5
            单笔交易最大持仓天数。示例: 5 表示持满 5 个交易日必须平仓
        min_lot_guarantee : bool, default=True
            一手保底机制。若风险预算计算股数经每手截断为 0，但在资金和仓位上限允许时保底买入 1 手。
        """
        profile = self.profile
        code_tag = f"[{self.current_stock_code}] " if getattr(self, 'current_stock_code', None) else ""
        cs = profile.currency_symbol

        # 初始化执行层组件：风险预算定仓器、出场状态机与仿真经纪商
        broker = SimulatedBroker(
            profile=profile,
            initial_capital=self.initial_capital,
            slippage_rate=self.slippage_rate,
        )
        sizer = RiskBudgetSizer(
            risk_per_trade=risk_per_trade,
            max_position_pct=max_position_pct,
            k_stop=k_stop,
            k_trail=k_trail,
            min_stop=min_stop,
            max_stop=max_stop,
            consecutive_loss_cut=consecutive_loss_cut,
            min_lot_guarantee=min_lot_guarantee,
        )
        exit_sm = ExitStateMachine(
            hard_stop_loss=hard_stop_loss,
            trailing_stop=trailing_stop,
            trailing_stop_min_profit=trailing_stop_min_profit,
            max_holding_days=max_holding_days,
            intraday_stop=intraday_stop,
        )

        position = broker.get_position(self.current_stock_code or profile.market)
        consecutive_losses = 0
        pending_close_reason = None
        pending_close_price = None

        records = []
        rows = list(df.itertuples(index=True))

        for idx, row in enumerate(rows):
            date = row.Index
            open_price = getattr(row, 'open', row.close)
            if pd.isna(open_price):
                open_price = row.close
            close_price = row.close
            low_price = getattr(row, 'low', close_price)
            if pd.isna(low_price):
                low_price = close_price
            high_price = getattr(row, 'high', close_price)
            if pd.isna(high_price):
                high_price = close_price

            exec_signal = int(row.exec_signal)
            limit_down = int(getattr(row, 'open_limit_down', getattr(row, 'limit_down', 0)))
            limit_up = int(getattr(row, 'open_limit_up', getattr(row, 'limit_up', 0)))

            # ----------------------------------------------------------------
            # A. 执行上一交易日决定的平仓指令
            # ----------------------------------------------------------------
            if pending_close_reason and position.is_open:
                if limit_down:
                    logger.info(f"[{date.date()}] {code_tag}⛔ 开盘封跌停无法卖出，平仓指令顺延: 【{pending_close_reason}】")
                else:
                    fill = min(pending_close_price, open_price) if pending_close_price is not None else open_price
                    cash, fee, trade_pnl, pnl_amount = broker.execute_sell(
                        trade_date=date,
                        symbol=self.current_stock_code or profile.market,
                        position=position,
                        fill_price=fill,
                        reason=pending_close_reason,
                        holding_days=position.holding_bars,
                        stock_code_tag=code_tag,
                    )
                    consecutive_losses = consecutive_losses + 1 if trade_pnl <= 0 else 0
                    pending_close_reason = None
                    pending_close_price = None

            # ----------------------------------------------------------------
            # B. 开仓买入
            # ----------------------------------------------------------------
            if not position.is_open and exec_signal == 1 and not limit_up:
                buy_price = open_price * (1 + self.slippage_rate)
                atr_pct = getattr(row, 'atr_pct', np.nan)
                scale = float(getattr(row, 'exec_position_scale', 1.0) or 0.0)

                shares_to_buy, active_stop_pct, active_trail_pct, stop_price = sizer.compute_size(
                    portfolio_value=broker.cash,
                    cash=broker.cash,
                    buy_price=buy_price,
                    profile=profile,
                    atr_pct=atr_pct,
                    position_scale=scale,
                    consecutive_losses=consecutive_losses,
                    hard_stop_loss=hard_stop_loss,
                    trailing_stop=trailing_stop,
                    trade_date=date,
                    stock_code=self.current_stock_code or "",
                )

                if shares_to_buy >= profile.lot_size:
                    target_horizon = int(getattr(row, 'exec_target_horizon', max_holding_days) or max_holding_days)
                    pos = broker.execute_buy(
                        trade_date=date,
                        symbol=self.current_stock_code or profile.market,
                        open_price=open_price,
                        shares_to_buy=shares_to_buy,
                        target_horizon=target_horizon,
                        active_stop_pct=active_stop_pct,
                        stop_price=stop_price,
                        bar_idx=idx,
                        close_price=close_price,
                        low_price=low_price,
                        active_trail_pct=active_trail_pct,
                        stock_code_tag=code_tag,
                    )
                    if pos:
                        position = pos
            elif not position.is_open and exec_signal == 1 and limit_up:
                logger.info(f"[{date.date()}] {code_tag}⛔ 开盘封涨停无法买入，放弃本次开仓信号")

            # ----------------------------------------------------------------
            # C. 持仓盯市与离场决策
            # ----------------------------------------------------------------
            if position.is_open:
                position.holding_bars += 1
                position.peak_price = max(position.peak_price, high_price)
                position.trough_price = min(position.trough_price, low_price) if position.trough_price > 0 else low_price

                # 1. 盘中止损检查
                should_exit_intraday, intraday_reason, intraday_fill = exit_sm.check_intraday_exit(
                    row=row,
                    position=position,
                    profile=profile,
                    idx=idx,
                    limit_down=limit_down,
                    stock_code_tag=code_tag,
                )

                if should_exit_intraday:
                    cash, fee, trade_pnl, pnl_amount = broker.execute_sell(
                        trade_date=date,
                        symbol=self.current_stock_code or profile.market,
                        position=position,
                        fill_price=intraday_fill,
                        reason=intraday_reason,
                        holding_days=position.holding_bars,
                        stock_code_tag=code_tag,
                    )
                    consecutive_losses = consecutive_losses + 1 if trade_pnl <= 0 else 0
                    pending_close_reason = None
                    pending_close_price = None
                    records.append({
                        'date': date,
                        'portfolio_value': broker.cash,
                        'cash': broker.cash,
                        'shares': 0,
                        'close': close_price,
                        'exec_signal': exec_signal,
                    })
                    continue
                elif intraday_reason:
                    pending_close_reason = intraday_reason
                    pending_close_price = intraday_fill

                # 2. 收盘 EOD 离场评估
                if not pending_close_reason:
                    pending_reason, pending_px = exit_sm.evaluate_eod_exit(
                        row=row,
                        position=position,
                        profile=profile,
                        idx=idx,
                        stock_code_tag=code_tag,
                    )
                    if pending_reason:
                        pending_close_reason = pending_reason
                        pending_close_price = pending_px
            else:
                logger.debug(f"[{date.date()}] {code_tag}⚪ 空仓观望 (exec_signal={exec_signal}, "
                             f"否决原因={getattr(row, 'veto_reason', '')})")

            # ----------------------------------------------------------------
            # D. 每日盯市净值结算
            # ----------------------------------------------------------------
            records.append({
                'date': date,
                'portfolio_value': broker.cash + position.shares * close_price,
                'cash': broker.cash,
                'shares': position.shares,
                'close': close_price,
                'exec_signal': exec_signal,
            })

        # 回测结束仍持仓则强制平仓
        if position.is_open and records:
            last_date = records[-1]['date']
            last_close = records[-1]['close']
            cash, fee, trade_pnl, pnl_amount = broker.execute_sell(
                trade_date=last_date,
                symbol=self.current_stock_code or profile.market,
                position=position,
                fill_price=last_close,
                reason='回测结束强平',
                holding_days=position.holding_bars,
                stock_code_tag=code_tag,
            )
            records[-1]['portfolio_value'] = broker.cash
            records[-1]['shares'] = 0

        trades = broker.trades
        logger.info(
            f"{code_tag}[状态机结算] 回测完成: 总交易笔数 {len([t for t in trades if t['action']=='SELL'])} 笔 | "
            f"期末总净值: {records[-1]['portfolio_value']:,.2f}"
        )

        backtest_df = pd.DataFrame(records).set_index('date')
        backtest_df['returns'] = backtest_df['portfolio_value'].pct_change().fillna(0.0)
        backtest_df['position_market_value'] = (backtest_df['shares'] * backtest_df['close']).round(2)
        backtest_df['position_pct'] = (backtest_df['position_market_value'] / backtest_df['portfolio_value']).round(4)
        cum_ret = (backtest_df['portfolio_value'] / self.initial_capital) - 1.0
        backtest_df['cum_return'] = cum_ret.round(6)
        hwm = backtest_df['portfolio_value'].cummax()
        backtest_df['drawdown'] = ((backtest_df['portfolio_value'] - hwm) / hwm).round(6)
        if len(backtest_df) > 0 and 'close' in backtest_df.columns:
            first_close = backtest_df['close'].iloc[0]
            backtest_df['benchmark_return'] = backtest_df['close'].pct_change().fillna(0.0).round(6)
            backtest_df['benchmark_cum_return'] = ((backtest_df['close'] / first_close) - 1.0).round(6)
            backtest_df['excess_return'] = (backtest_df['cum_return'] - backtest_df['benchmark_cum_return']).round(6)

        return backtest_df, trades


    def run_complete_backtest(
        self,
        stock_code: str,
        output_dir: str,
        z_score_threshold: float = 1.2,
        volatility_window: int = 14,
        hard_stop_loss: float = 0.04,
        trailing_stop: float = 0.025,
        max_holding_days: int = 5,
        show_plot: bool = False,
        entry_percentile: float = 0.80,
        calib_window: int = 60,
        dynamic_calib: bool = True,
        calib_window_min: int = 20,
        vol_high_thresh: float = 0.40,
        vol_low_thresh: float = 0.20,
        risk_per_trade: float = 0.01,
        max_position_pct: float = 0.80,
        min_lot_guarantee: bool = True,
        intraday_stop: bool = True,
        require_skill: bool = False,
        market: str = None,
        forecast_horizon: str = None,
        max_auto_horizon: int = 10,
        particle_filter: bool = True,
        pf_confidence_threshold: float = 0.35,
        reversal_aware_debias: bool = True,
        reversal_debias_weight: float = 0.0,
        enable_log: bool = True,
        clear_log: bool = False,
        **kwargs,
    ) -> tuple:
        """
        一键完整回测管线：市场规则 → 加载 → 技能体检 → 对齐/信号 → 撮合 → 指标 → 绘图

        require_skill=True 时，预测技能体检未通过将直接中止回测，
        避免在没有预测能力的标的上继续调参。

        返回: (metrics, signal_df, backtest_df, trades, fig)
        """
        print(f"\n{'='*60}")
        print(f"🎯  Kronos Pro 回测引擎  —  {stock_code}")
        print(f"{'='*60}")

        try:
            if clear_log:
                clear_logs(log_dir=output_dir, stock_code=stock_code)

            print("\n[Step 1] 设定市场规则 & 加载数据")
            # 依股票代码切换市场规则（整手、印花税、T+N、涨跌停、结算币种）
            self.set_market(stock_code, market=market, log_dir=output_dir, enable_log=enable_log)
            clean_code = getattr(self, 'current_stock_code', stock_code)
            print(f"   市场规则: {self.profile.describe()}")

            logger.info(f"[{clean_code}] ==================== KRONOS QUANT BACKTEST START ====================")
            logger.info(f"[{clean_code}] 标的代码: {stock_code} | 市场规则: {self.profile.describe()} | 初始资金: {self.profile.currency_symbol}{self.initial_capital:,.2f}")
            logger.info(f"[{clean_code}] 策略配置: 止损={hard_stop_loss:.2%}, 移动止盈={trailing_stop:.2%}, 最大持仓={max_holding_days}天, 动态校准={'开启' if dynamic_calib else '关闭'}, 风险预算={risk_per_trade:.2%}")

            hist_df, pred_df = self.load_data(stock_code, particle_filter=particle_filter)

            print("\n[Step 2] 时序对齐 & 信号生成")
            signal_df = self.align_and_generate_signals(
                hist_df, pred_df,
                z_score_threshold=z_score_threshold,
                volatility_window=volatility_window,
                forecast_horizon=forecast_horizon,
                hard_stop_loss=hard_stop_loss,
                entry_percentile=entry_percentile,
                calib_window=calib_window,
                dynamic_calib=dynamic_calib,
                calib_window_min=calib_window_min,
                vol_high_thresh=vol_high_thresh,
                vol_low_thresh=vol_low_thresh,
                max_auto_horizon=max_auto_horizon,
                particle_filter=particle_filter,
                pf_confidence_threshold=pf_confidence_threshold,
                reversal_aware_debias=reversal_aware_debias,
                reversal_debias_weight=reversal_debias_weight,
            )

            print("\n[Step 2b] 预测技能体检")
            skill = self.evaluate_prediction_skill(signal_df, hard_stop_loss=hard_stop_loss)
            self.print_skill_report(skill, stock_code)
            if require_skill and not skill.get('tradable', False):
                reason = skill.get('error') or '; '.join(skill.get('不通过原因', []))
                raise ValueError(
                    f"\n[准入拦截] {stock_code} 未通过预测技能体检，已中止回测。\n"
                    f"  原因: {reason}\n"
                    f"  说明: 预测无排序能力时，任何参数组合的正收益都只是路径运气。\n"
                    f"  如需强行回测，请设置 require_skill=False。"
                )

            print("\n[Step 3] 逐日撮合状态机")
            backtest_df, trades = self.run_backtest_engine(
                signal_df,
                hard_stop_loss=hard_stop_loss,
                trailing_stop=trailing_stop,
                max_holding_days=max_holding_days,
                min_lot_guarantee=min_lot_guarantee,
                risk_per_trade=risk_per_trade,
                max_position_pct=max_position_pct,
                intraday_stop=intraday_stop,
            )

            print("\n[Step 4] 绩效指标计算")
            metrics = self.evaluate_metrics(backtest_df, trades)
            metrics['_skill'] = skill

            print("\n[Step 5] 生成图表")
            fig = self.plot_backtest_results(
                signal_df, backtest_df, metrics, trades,
                stock_code, output_dir, show_plot=show_plot,
            )

            print("\n[Step 6] 导出量化分析数据集")
            params = {
                'z_score_threshold': z_score_threshold,
                'volatility_window': volatility_window,
                'hard_stop_loss': hard_stop_loss,
                'trailing_stop': trailing_stop,
                'max_holding_days': max_holding_days,
                'entry_percentile': entry_percentile,
                'calib_window': calib_window,
                'dynamic_calib': dynamic_calib,
                'calib_window_min': calib_window_min,
                'vol_high_thresh': vol_high_thresh,
                'vol_low_thresh': vol_low_thresh,
                'risk_per_trade': risk_per_trade,
                'max_position_pct': max_position_pct,
                'min_lot_guarantee': min_lot_guarantee,
                'intraday_stop': intraday_stop,
                'require_skill': require_skill,
                'forecast_horizon': forecast_horizon,
                'particle_filter': particle_filter,
                'pf_confidence_threshold': pf_confidence_threshold,
            }
            self.export_backtest_artifacts(
                stock_code=stock_code,
                output_dir=output_dir,
                signal_df=signal_df,
                backtest_df=backtest_df,
                trades=trades,
                metrics=metrics,
                params=params,
            )

            self._print_report(stock_code, metrics, trades)
            logger.info(f"[{clean_code}] ==================== KRONOS QUANT BACKTEST COMPLETED ====================\n")
            return metrics, signal_df, backtest_df, trades, fig

        except ValueError as e:
            print(str(e))
            return None, None, None, None, None
        except Exception as e:
            import traceback
            print(f"\n❌ 回测异常: {e}")
            traceback.print_exc()
            return None, None, None, None, None
        finally:
            self.cleanup_logger()

    def export_backtest_artifacts(
        self,
        stock_code: str,
        output_dir: str,
        signal_df: pd.DataFrame,
        backtest_df: pd.DataFrame,
        trades: list,
        metrics: dict,
        params: dict = None,
    ) -> dict:
        """
        全量导出量化回测结构化数据产物，用于后续精细化量化分析与策略复盘：
        1. {code}_trades.csv: 逐笔交易明细表 (开平仓价格/股数/税费/收益率/绝对盈亏/持仓天数/出场原因/MFE/MAE)
        2. {code}_daily_equity.csv: 逐日盯市资产净值与仓位暴露表
        3. {code}_daily_backtest.csv: 兼容历史看板命名的逐日净值表
        4. {code}_signals.csv: 逐日预测价、动量、Z-Score、动态阈值、执行信号与否决原因诊断表
        5. {code}_metrics.json: 完整绩效评估指标、技能评分与超参数快照
        """
        clean_code, _, _ = SymbolResolver.resolve(stock_code)
        clean_code = clean_code or stock_code
        os.makedirs(output_dir, exist_ok=True)
        artifacts = {}

        # 1. 逐笔交易记录
        trades_file = os.path.join(output_dir, f"{clean_code}_trades.csv")
        if trades:
            tdf = pd.DataFrame(trades)
        else:
            tdf = pd.DataFrame(columns=[
                'date', 'action', 'price', 'shares', 'commission', 'pnl_pct',
                'pnl_amt', 'holding_days', 'reason', 'cash_balance', 'entry_date',
                'entry_price', 'mfe_pct', 'mae_pct', 'symbol'
            ])
        tdf.to_csv(trades_file, index=False, encoding='utf-8-sig')
        artifacts['trades'] = trades_file

        # 2. 逐日资产净值表 (同时导出 daily_equity.csv 与 daily_backtest.csv 保持完全兼容)
        equity_file = os.path.join(output_dir, f"{clean_code}_daily_equity.csv")
        backtest_file = os.path.join(output_dir, f"{clean_code}_daily_backtest.csv")
        if backtest_df is not None and not backtest_df.empty:
            backtest_df.to_csv(equity_file, encoding='utf-8-sig')
            backtest_df.to_csv(backtest_file, encoding='utf-8-sig')
        artifacts['equity'] = equity_file
        artifacts['daily_backtest'] = backtest_file

        # 3. 逐日预测与信号诊断表
        signals_file = os.path.join(output_dir, f"{clean_code}_signals.csv")
        if signal_df is not None and not signal_df.empty:
            signal_df.to_csv(signals_file, encoding='utf-8-sig')
        artifacts['signals'] = signals_file

        # 4. 指标与超参数 JSON
        metrics_file = os.path.join(output_dir, f"{clean_code}_metrics.json")
        m_clean = {}
        skill_clean = {}
        if metrics:
            for k, v in metrics.items():
                if k == '_skill' and isinstance(v, dict):
                    skill_clean = {sk: (float(sv) if isinstance(sv, (np.floating, float, int, np.integer)) else sv)
                                   for sk, sv in v.items()}
                elif not k.startswith('_'):
                    if isinstance(v, (np.floating, float)):
                        m_clean[k] = None if (np.isnan(v) or np.isinf(v)) else float(v)
                    elif isinstance(v, (np.integer, int)):
                        m_clean[k] = int(v)
                    elif isinstance(v, bool):
                        m_clean[k] = bool(v)
                    else:
                        m_clean[k] = str(v)

        clean_params = {}
        if params:
            for pk, pv in params.items():
                if isinstance(pv, (str, int, float, bool, list, dict)) or pv is None:
                    clean_params[pk] = pv
                else:
                    clean_params[pk] = str(pv)

        payload = {
            'stock_code': stock_code,
            'clean_code': clean_code,
            'market': self.profile.market,
            'currency': self.profile.currency,
            'currency_symbol': self.profile.currency_symbol,
            'initial_capital': self.initial_capital,
            'commission_rate': self.profile.commission_rate,
            'stamp_duty_rate': self.profile.stamp_duty_sell,
            'slippage_rate': self.slippage_rate,
            'exported_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'strategy_params': clean_params,
            'metrics': m_clean,
            'skill_evaluation': skill_clean,
        }
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        artifacts['metrics'] = metrics_file

        logger.info(f"[{clean_code}] 📦 量化回测数据已全部导出为结构化分析文件:")
        logger.info(f"[{clean_code}]   - 逐笔成交流水: {trades_file}")
        logger.info(f"[{clean_code}]   - 逐日资产净值: {equity_file}")
        logger.info(f"[{clean_code}]   - 预测信号明细: {signals_file}")
        logger.info(f"[{clean_code}]   - 绩效指标参数: {metrics_file}")
        return artifacts

    def _print_report(self, stock_code: str, metrics: dict, trades: list):
        """控制台输出格式化回测报告"""
        print(f"\n{'='*60}")
        print(f"📊  {stock_code} 回测报告  [样本外 Backtest 模式]")
        print(f"{'='*60}")
        fmt_pct = {'累计收益率', '年化收益率', '年化波动率', '最大回撤', '胜率',
                   '单笔平均盈亏', '持仓占比', '最大单笔盈利', '最大单笔亏损', '平均MFE', '平均MAE'}
        cur = self.profile.currency
        cs = self.profile.currency_symbol
        for k, v in metrics.items():
            if k.startswith('_'):
                continue
            if isinstance(v, bool):
                print(f"  {k:<12}: {'是' if v else '否'}")
            elif v is None or (isinstance(v, float) and not np.isfinite(v)):
                # 样本不足时年化/卡玛等指标本就不该给数，显式标注而非填 0
                label = '不适用' if k in ('年化收益率', '卡玛比率') and not metrics.get('可年化', True) else 'N/A'
                if isinstance(v, float) and v == float('inf'):
                    label = '∞ (无亏损样本)'
                print(f"  {k:<12}: {label}")
            elif k in fmt_pct:
                print(f"  {k:<12}: {v:>+.2%}")
            elif k in ('总交易笔数', '期末强平笔数', '回测天数'):
                print(f"  {k:<12}: {int(v)}")
            elif k in ('期末总资产', '总手续费', '总盈亏金额'):
                print(f"  {k:<12}: {cur} {v:>,.2f}")
            elif k == '平均持仓天数':
                print(f"  {k:<12}: {v:.1f} 天")
            else:
                print(f"  {k:<12}: {v:.4f}")

        if not metrics.get('统计可靠', True):
            print(f"\n  ⚠️ 交易样本仅 {metrics.get('总交易笔数', 0)} 笔（<20），"
                  f"胜率与盈亏比不具统计意义，不可作为参数选择依据。")
        if not metrics.get('可年化', True):
            print(f"  ⚠️ 回测仅 {metrics.get('回测天数', 0)} 天（<120），"
                  f"年化与卡玛比率已标记为不适用，短样本外推会严重失真。")

        sell_t = [t for t in trades if t['action'] == 'SELL']
        if sell_t:
            print(f"\n  平仓原因分布:")
            reason_counts = {}
            for t in sell_t:
                reason_counts[t['reason']] = reason_counts.get(t['reason'], 0) + 1
            for reason, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
                print(f"    {reason}: {cnt} 次")

        print(f"\n  近期交易流水 (最多10条):")
        for t in trades[-10:]:
            pnl_str = f"  盈亏: {t['pnl_pct']:+.2%}" if t['action'] == 'SELL' else ""
            if t['action'] == 'SELL' and 'pnl_amt' in t:
                pnl_str += f" ({cs}{t['pnl_amt']:+,.2f})"
            hold_str = f" [持仓{t.get('holding_days', 0)}天]" if t['action'] == 'SELL' and t.get('holding_days') else ""
            print(f"    {t['date'].strftime('%Y-%m-%d')}  "
                  f"{t['action']:<4} {t['shares']:>6}股 @ {cs}{t['price']:.3f}"
                  f"{pnl_str}{hold_str}")
        print(f"{'='*60}\n")




# 引擎类通用别名
BacktestEngine = KronosBacktesterPro
