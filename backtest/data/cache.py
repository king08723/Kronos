"""
backtest/data/cache.py
回测预测数据缓存指纹与特征规范化工具
通过参数指纹 (SHA / JSON) 严格防止因模型参数变更导致的数据污染与未来函数泄漏。
"""

from __future__ import annotations

import os
import json
from typing import Dict, Any, Tuple
import pandas as pd


class CacheManager:
    """预测缓存与指纹管理器"""

    @staticmethod
    def build_fingerprint(**kwargs) -> Dict[str, Any]:
        """生成预测缓存的参数指纹；任一生成参数变化都会导致指纹不匹配。"""
        fp = {
            k: (str(v) if not isinstance(v, (int, float, bool)) else v)
            for k, v in kwargs.items()
        }
        fp['schema_version'] = 2  # v2 = 修正后的列结构与粒子滤波支持
        return fp

    @staticmethod
    def fingerprint_path(csv_path: str) -> str:
        return csv_path + '.meta.json'

    @classmethod
    def write_cache_fingerprint(cls, csv_path: str, fingerprint: Dict[str, Any]) -> None:
        try:
            with open(cls.fingerprint_path(csv_path), 'w', encoding='utf-8') as f:
                json.dump(fingerprint, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"   ⚠️ 写入缓存指纹失败（不影响预测结果）: {e}")

    @classmethod
    def check_cache_fingerprint(cls, csv_path: str, expected: Dict[str, Any]) -> Tuple[bool, str]:
        """
        校验缓存指纹。
        返回 (是否匹配, 不匹配原因描述)。缺失指纹文件视为不匹配，
        因为旧缓存的生成参数不可考，继续复用会造成无法归因的实验污染。
        """
        fp_path = cls.fingerprint_path(csv_path)
        if not os.path.exists(fp_path):
            return False, "缺少参数指纹文件（旧版缓存，生成参数不可考）"
        try:
            with open(fp_path, 'r', encoding='utf-8') as f:
                actual = json.load(f)
        except Exception as e:
            return False, f"指纹文件读取失败: {e}"

        diffs = []
        for k, v in expected.items():
            if k not in actual:
                diffs.append(f"{k}: 缺失 -> {v}")
            elif str(actual[k]) != str(v):
                diffs.append(f"{k}: {actual[k]} -> {v}")
        if diffs:
            return False, "参数已变更 (" + ", ".join(diffs) + ")"
        return True, ""

    @staticmethod
    def normalize_pred_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        统一预测文件列名与字段类型，兼容各种历史版本及粒子滤波衍生列。
        """
        out = df.copy()
        # 日期列规范化
        for c in ('pred_date', 'date', 'predict_date', 'target_date'):
            if c in out.columns:
                out['date'] = pd.to_datetime(out[c]).dt.date
                break

        # 预测特征别名映射
        col_mappings = {
            'predicted_price': 'pred_close_mean',
            'pred_mean': 'pred_close_mean',
            'pred_close': 'pred_close_mean',
            'pred_median': 'pred_close_median',
            'predicted_peak': 'pred_peak_median',
            'pred_peak': 'pred_peak_median',
            'predicted_trough': 'pred_trough_median',
            'pred_trough': 'pred_trough_median',
            'predicted_min_down': 'pred_min_down_quantile',
            'predicted_max_up': 'pred_max_up_quantile',
            'predicted_target_horizon': 'pred_target_horizon',
            'target_horizon': 'pred_target_horizon',
        }
        for old_col, new_col in col_mappings.items():
            if old_col in out.columns and new_col not in out.columns:
                out[new_col] = out[old_col]

        # 数值类型安全转换
        numeric_cols = [
            'pred_close_mean', 'pred_close_median', 'pred_peak_median',
            'pred_trough_median', 'pred_min_down_quantile', 'pred_max_up_quantile',
            'pred_target_horizon', 'n_samples', 'is_interpolated',
            'pf_confidence', 'pf_val_error', 'pf_effective_n',
        ]
        for col in numeric_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors='coerce')

        return out
