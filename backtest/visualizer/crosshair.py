"""
backtest/visualizer/crosshair.py
Streamlit 看板全图贯通垂直十字光标 (Continuous Crosshair Timeline across subplots)
"""

from __future__ import annotations


def inject_continuous_crosshair_script():
    """
    注入全图贯通的垂直十字光标时间标尺 (Continuous Crosshair Timeline across 4 subplots)
    使 4 联子图在鼠标悬停移动时共用一条无断点的垂直虚线参考线，贯穿全部 4 个子图。
    """
    import streamlit as st
    html_code = """
    <script>
    (function() {
        const doc = window.parent.document || document;
        
        function setupCrosshair() {
            const chartDivs = doc.querySelectorAll('.stPlotlyChart');
            chartDivs.forEach(chartContainer => {
                if (chartContainer.dataset.crosshairInit === 'true') return;
                
                const plotlyEl = chartContainer.querySelector('.js-plotly-plot');
                if (!plotlyEl) return;
                
                chartContainer.dataset.crosshairInit = 'true';
                chartContainer.style.position = 'relative';

                // 创建全局贯通的垂直时间虚线标尺
                let vline = chartContainer.querySelector('.quant-crosshair-vline');
                if (!vline) {
                    vline = doc.createElement('div');
                    vline.className = 'quant-crosshair-vline';
                    vline.style.position = 'absolute';
                    vline.style.top = '30px';
                    vline.style.bottom = '35px';
                    vline.style.width = '0px';
                    vline.style.borderLeft = '1.5px dashed #E74C3C';
                    vline.style.pointerEvents = 'none';
                    vline.style.zIndex = '9999';
                    vline.style.display = 'none';
                    vline.style.boxShadow = '0 0 3px rgba(231, 76, 60, 0.4)';
                    chartContainer.appendChild(vline);
                }

                // 监听鼠标在图表区域内的移动
                chartContainer.addEventListener('mousemove', function(e) {
                    const rect = chartContainer.getBoundingClientRect();
                    const mouseX = e.clientX - rect.left;
                    
                    // 尝试从 Plotly 内部 SVG 捕捉精准吸附的 xspikeline 坐标
                    let targetX = mouseX;
                    const spike = chartContainer.querySelector('line.xspikeline, line.spikeline');
                    if (spike) {
                        const sx = parseFloat(spike.getAttribute('x1'));
                        if (!isNaN(sx) && sx > 0) {
                            targetX = sx;
                        }
                    }

                    // 绘图区域边界检测 (留出左 Y 轴刻度 58px 与 右边距 35px)
                    const minX = 58;
                    const maxX = rect.width - 35;
                    if (targetX >= minX && targetX <= maxX) {
                        vline.style.left = targetX + 'px';
                        vline.style.display = 'block';
                    } else {
                        vline.style.display = 'none';
                    }
                });

                chartContainer.addEventListener('mouseleave', function() {
                    vline.style.display = 'none';
                });
            });
        }

        setupCrosshair();
        if (!window._quantCrosshairTimer) {
            window._quantCrosshairTimer = setInterval(setupCrosshair, 400);
        }
    })();
    </script>
    """
    try:
        st.html(html_code, unsafe_allow_javascript=True)
    except Exception:
        try:
            import streamlit.components.v1 as components
            components.html(html_code, height=0)
        except Exception:
            pass
