"""
backtest/engine/skill_evaluator.py
AI 预测能力前置体检评价体系 (Prediction Skill Evaluation)
包含：
- Spearman 秩相关信息系数 (IC) 与 t 显著性检验
- 方向命中率 (Hit Rate)
- 预测收益与真实收益幅度比 (Magnitude Ratio)
- 极端下探跌破率 (Breach Rate)
- 粒子滤波置信度与验证误差评价
- 严密准入门禁拦截 (Tradable Gatekeeper)
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class SkillEvaluator:
    """AI 预测技能前置体检评价器"""

    @staticmethod
    def evaluate(df: pd.DataFrame, hard_stop_loss: float = 0.04) -> Dict[str, Any]:
        """
        在回测之前体检预测本身有没有可用的预测能力。
        策略再精巧也救不回一个没有技能的预测源，所以把「能不能交易这个标的」
        变成一道显式的前置检查。
        """
        out = {}
        need = {'pred_close', 'close', 'close_prev'}
        if not need.issubset(df.columns) or len(df) < 20:
            return {'error': f'样本不足或缺列，无法体检（行数={len(df)}）', 'tradable': False}

        realized = ((df['close'] - df['close_prev']) / df['close_prev']).astype(float)
        # 优先使用单日预期收益评测单日基础选时能力；若无则对标时界期望
        if 'pred_close_1d' in df.columns and df['pred_close_1d'].notna().all():
            predicted = ((df['pred_close_1d'] - df['close_prev']) / df['close_prev']).astype(float)
            horizon_desc = "1日预期"
        else:
            predicted = ((df['pred_close'] - df['close_prev']) / df['close_prev']).astype(float)
            horizon_desc = "时界期望"
        out['评估口径'] = horizon_desc

        mask = realized.notna() & predicted.notna()
        realized, predicted = realized[mask], predicted[mask]
        n = len(realized)
        out['样本数'] = n
        if n < 20:
            return {'error': f'有效样本仅 {n} 条', 'tradable': False}

        # Spearman 秩相关
        ic = float(predicted.rank().corr(realized.rank()))
        out['IC'] = ic
        if abs(ic) < 1 - 1e-12 and n > 2:
            t_stat = ic * np.sqrt((n - 2) / (1 - ic ** 2))
        else:
            t_stat = 0.0
        out['IC_t值'] = float(t_stat)
        out['IC显著'] = bool(abs(t_stat) >= 1.96)

        # 方向命中率
        out['方向命中率'] = float(((predicted > 0) == (realized > 0)).mean())

        # 幅度校准
        mean_pred, mean_real = float(predicted.mean()), float(realized.mean())
        out['预测收益均值'] = mean_pred
        out['真实收益均值'] = mean_real
        out['幅度比'] = (mean_pred / mean_real) if abs(mean_real) > 1e-9 else float('inf')
        out['幅度偏差'] = mean_pred - mean_real

        # 下探预测校准
        if 'pred_min_down' in df.columns and 'low' in df.columns:
            sub = df.loc[mask]
            actual_down = ((sub['low'] - sub['close_prev']) / sub['close_prev']).astype(float)
            pmd = sub['pred_min_down'].astype(float)
            valid = actual_down.notna() & pmd.notna()
            if valid.sum() >= 20:
                out['下探跌破率'] = float((actual_down[valid] < pmd[valid]).mean())

        # 粒子滤波置信度指标
        if 'pf_confidence' in df.columns:
            out['粒子滤波平均置信度'] = float(df['pf_confidence'].mean())
        if 'pf_val_error' in df.columns:
            out['粒子滤波验证误差'] = float(df['pf_val_error'].mean())

        # 准入判定
        reasons = []
        if abs(ic) < 0.03:
            reasons.append(f'IC={ic:.3f} 接近噪声(<0.03)')
        if not out['IC显著']:
            reasons.append(f'IC 不显著(t={t_stat:.2f})')
        if ic < 0:
            reasons.append(f'IC 为负({ic:.3f})，预测方向与真实相反')
        out['不通过原因'] = reasons
        out['tradable'] = len(reasons) == 0
        return out

    @staticmethod
    def print_report(skill: Dict[str, Any], stock_code: str = ''):
        """打印预测技能体检报告"""
        print(f"\n{'='*64}")
        print(f"  预测技能体检 {stock_code}")
        print(f"{'='*64}")
        if 'error' in skill:
            print(f"  ❌ {skill['error']}")
            return
        if '评估口径' in skill:
            print(f"  评估口径          : {skill['评估口径']} 对标 1日真实收益")
        print(f"  样本数            : {skill['样本数']}")
        print(f"  IC (秩相关)       : {skill['IC']:+.4f}  (t={skill['IC_t值']:+.2f}, "
              f"{'显著' if skill['IC显著'] else '不显著'})")
        print(f"  方向命中率        : {skill['方向命中率']:.1%}")
        print(f"  预测收益均值      : {skill['预测收益均值']:+.3%}")
        print(f"  真实收益均值      : {skill['真实收益均值']:+.3%}")
        mr = skill['幅度比']
        print(f"  幅度比(预测/真实) : {mr:.1f}x" if np.isfinite(mr) else "  幅度比(预测/真实) : inf")
        if '粒子滤波平均置信度' in skill:
            print(f"  粒子滤波平均置信度: {skill['粒子滤波平均置信度']:.1%}")
        if '粒子滤波验证误差' in skill:
            print(f"  粒子滤波验证误差  : {skill['粒子滤波验证误差']:.2%}")
        if '下探跌破率' in skill:
            br = skill['下探跌破率']
            flag = '✅' if br <= 0.20 else ('⚠️' if br <= 0.35 else '❌')
            print(f"  下探跌破率        : {br:.1%} {flag}  (目标≈10%)")
        if skill['tradable']:
            print(f"  结论              : ✅ 通过准入，可进入回测")
        else:
            print(f"  结论              : ❌ 未通过准入")
            for r in skill['不通过原因']:
                print(f"       - {r}")
        print(f"{'='*64}")
