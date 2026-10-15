const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

let hljs;
try {
  hljs = require('/tmp/kronos_html_builder/node_modules/highlight.js');
} catch (e) {
  try {
    hljs = require('highlight.js');
  } catch (e2) {
    console.warn('highlight.js not found, fallback to basic escaping');
  }
}

const MD_PATH = '/Users/jyy/Documents/github_projects/Kronos/backtest/run_backtest_kronos_pro readme.md';
const TEMP_HTML = '/tmp/marked_full_out.html';
const OUT_PATH = '/Users/jyy/Documents/github_projects/Kronos/backtest/run_backtest_kronos_pro_readme.html';

console.log('1. Converting full Markdown file with npx marked -o...');
execSync(`npx -y marked -i "${MD_PATH}" -o "${TEMP_HTML}" --gfm`, {
  maxBuffer: 50 * 1024 * 1024,
  encoding: 'utf-8'
});

const rawHtml = fs.readFileSync(TEMP_HTML, 'utf-8');
console.log('Raw HTML size:', (rawHtml.length / 1024).toFixed(1), 'KB, chars:', rawHtml.length);

// Helper to decode HTML entities
function decodeEntities(str) {
  return str
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}

// Custom Highlighters
function highlightBashCode(raw) {
  const lines = raw.split('\n');
  return lines.map(line => {
    if (line.trim().startsWith('#')) {
      return `<span class="hljs-comment">${line}</span>`;
    }
    return line
      .replace(/^(\s*)(python|streamlit|npx|node|bash|sh|open)\b/g, '$1<span class="cmd-name">$2</span>')
      .replace(/(--[a-zA-Z0-9\-_]+|-auto)/g, '<span class="cmd-flag">$1</span>')
      .replace(/\b([0-9]{6}|[0-9]{5}|AAPL|TSLA|MSFT|NVDA)\b/g, '<span class="cmd-arg">$1</span>')
      .replace(/(backtest\/[a-zA-Z0-9_\/]+\.(?:py|csv|html|pdf|png|json))/g, '<span class="cmd-path">$1</span>');
  }).join('\n');
}

function highlightRatingReport(text) {
  return text
    .replace(/(★+☆*)/g, '<span class="term-stars">$1</span>')
    .replace(/(\[\d\]\s*[^:(]+)/g, '<span class="term-dim-title">$1</span>')
    .replace(/(卓越稳健级)/g, '<span class="term-tier-badge">$1</span>')
    .replace(/(Institutional Tier-1 Quant Profile)/g, '<span class="term-tier-sub">$1</span>')
    .replace(/(KRONOS PRO QUANT STRATEGY RATING REPORT)/g, '<span class="term-main-title">$1</span>');
}

function highlightMathFormulas(raw) {
  const lines = raw.split('\n');
  return lines.map(line => {
    let l = line;
    // Comments or text explanations in parentheses like (即 2.15%)
    l = l.replace(/(\([^)]*[\u4e00-\u9fa5]+[^)]*\))/g, '<span class="calc-note">$1</span>');
    // Math functions
    l = l.replace(/\b(sqrt|mean|std|clip|max|min|count|sum|valid|sign|abs|rolling|np\.where)\b/g, '<span class="calc-func">$1</span>');
    // Key indicators & variables
    l = l.replace(/\b(vol_14d|volatility_scaled|atr|atr_pct|close_prev|expected_return|expected_return_adj|signal_score|risk_budget|stop_distance|stop_price|shares_to_buy|buy_price|sell_price|portfolio_value|cash|total_return|sharpe|ESS|weights|pf_confidence|target_horizon|optimal_horizon|pred_close|pred_path_max|pred_min_down|pred_slope|position_scale|pred_bias|pred_bias_effective|reversal_mask|bias_weight|avg_mfe|avg_mae|exposure|stat_reliable|forced_exits|daily_volatility|annual_volatility|turnover_rate)\b/g, '<span class="calc-var">$1</span>');
    // Numbers
    l = l.replace(/([+−\-]?\b\d+(?:\.\d+)?%?)\b/g, '<span class="calc-num">$1</span>');
    return l;
  }).join('\n');
}

function formatCodeBlock(rawCode, lang) {
  let highlighted = '';
  const cleanLang = (lang || '').toLowerCase();

  if (cleanLang === 'bash' || cleanLang === 'sh') {
    highlighted = highlightBashCode(rawCode);
  } else if (cleanLang === 'python' || cleanLang === 'py') {
    if (hljs) {
      highlighted = hljs.highlight(rawCode, { language: 'python', ignoreIllegals: true }).value;
    } else {
      highlighted = rawCode.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
  } else if (cleanLang === 'json') {
    if (hljs) {
      highlighted = hljs.highlight(rawCode, { language: 'json', ignoreIllegals: true }).value;
    } else {
      highlighted = rawCode.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
  } else {
    // Text block
    if (rawCode.includes('KRONOS PRO QUANT STRATEGY RATING REPORT')) {
      highlighted = highlightRatingReport(rawCode);
    } else if (rawCode.includes('=') && (rawCode.includes('sqrt') || rawCode.includes('std') || rawCode.includes('atr') || rawCode.includes('portfolio_value') || rawCode.includes('mean') || rawCode.includes('np.where'))) {
      highlighted = highlightMathFormulas(rawCode);
    } else if (hljs) {
      highlighted = hljs.highlightAuto(rawCode).value;
    } else {
      highlighted = rawCode.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
  }

  const lines = highlighted.split('\n');
  if (lines.length > 1 && lines[lines.length - 1].trim() === '') {
    lines.pop();
  }

  const lineCount = lines.length;
  const numberedLines = lines.map((lineContent, idx) => {
    const lineNum = idx + 1;
    return `<div class="code-line"><span class="line-num" data-line="${lineNum}">${lineNum}</span><span class="line-text">${lineContent || ' '}</span></div>`;
  }).join('\n');

  const langLabel = cleanLang ? cleanLang.toUpperCase() : (rawCode.includes('KRONOS PRO QUANT') ? 'TERMINAL' : 'CODE');

  return `
<div class="code-card" data-raw="${encodeURIComponent(rawCode)}">
  <div class="code-header">
    <div class="code-dots">
      <span class="dot dot-red"></span>
      <span class="dot dot-yellow"></span>
      <span class="dot dot-green"></span>
    </div>
    <div class="code-meta-wrap">
      <span class="code-lang-badge lang-${cleanLang || 'text'}">${langLabel}</span>
      <span class="code-line-count">${lineCount} 行代码</span>
    </div>
    <button class="copy-code-btn" onclick="copyCodeBlock(this)">
      <svg class="copy-icon" viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
      <span>复制</span>
    </button>
  </div>
  <div class="code-viewport">
    <div class="code-lines">${numberedLines}</div>
  </div>
</div>`;
}

// 2. Post-processing
let processedHtml = rawHtml;

// Transform source links blockquote at top into hero cards
const sourceQuoteRegex = /<blockquote>\s*<p>统一调度门面：[\s\S]*?多周期与自定义策略范例：[\s\S]*?<\/p>\s*<\/blockquote>/;
const sourceLinksHtml = `
<div class="source-modules-grid">
  <div class="module-card">
    <div class="module-icon">⚡</div>
    <div class="module-info">
      <span class="module-title">统一调度门面 (Facade)</span>
      <a class="module-link" href="file:///Users/jyy/Documents/github_projects/Kronos/backtest/run_backtest_kronos_pro.py">run_backtest_kronos_pro.py</a>
      <span class="module-desc">标准门面入口，支持 CLI 全参数调度与单标的 Streamlit 深度看板</span>
    </div>
  </div>
  <div class="module-card">
    <div class="module-icon">🏛️</div>
    <div class="module-info">
      <span class="module-title">多市场全真规则内核</span>
      <a class="module-link" href="file:///Users/jyy/Documents/github_projects/Kronos/backtest/core/market_rules.py">core/market_rules.py</a>
      <span class="module-desc">A股 / 港股 / 美股 整手、分段印花税、佣金保底、滑点与涨跌停微观画像</span>
    </div>
  </div>
  <div class="module-card">
    <div class="module-icon">📦</div>
    <div class="module-info">
      <span class="module-title">股票池批量推演调度</span>
      <a class="module-link" href="file:///Users/jyy/Documents/github_projects/Kronos/backtest/run_backtest_batch.py">run_backtest_batch.py</a>
      <span class="module-desc">多标的股票池自动化编排、多进程并发推演与大盘排行榜自动生成</span>
    </div>
  </div>
  <div class="module-card">
    <div class="module-icon">📊</div>
    <div class="module-info">
      <span class="module-title">统一量化投研大盘</span>
      <a class="module-link" href="file:///Users/jyy/Documents/github_projects/Kronos/backtest/app_backtest_dashboard.py">app_backtest_dashboard.py</a>
      <span class="module-desc">Streamlit 池化筛选大盘 + 单股穿透深度下钻、多轴连续光标联动</span>
    </div>
  </div>
  <div class="module-card">
    <div class="module-icon">🧩</div>
    <div class="module-info">
      <span class="module-title">生产级模块化解耦体系</span>
      <a class="module-link" href="file:///Users/jyy/Documents/github_projects/Kronos/backtest/engine/">core · engine · strategies</a>
      <span class="module-desc">按职责完全解耦：执行层(sizer/broker)、数据层、策略层与可视化层</span>
    </div>
  </div>
  <div class="module-card">
    <div class="module-icon">⏱️</div>
    <div class="module-info">
      <span class="module-title">多周期与自定义策略范例</span>
      <a class="module-link" href="file:///Users/jyy/Documents/github_projects/Kronos/backtest/run_custom_strategy.py">run_custom_strategy.py</a>
      <span class="module-desc">支持 15m/30m/60m/1d 时间粒度、日内突破与可插拔策略基类扩展</span>
    </div>
  </div>
</div>
`;
if (sourceQuoteRegex.test(processedHtml)) {
  processedHtml = processedHtml.replace(sourceQuoteRegex, sourceLinksHtml);
} else {
  // Fallback for older quote regex
  const fallbackQuoteRegex = /<blockquote>\s*<p>对应源码：[\s\S]*?<\/p>\s*<\/blockquote>/;
  processedHtml = processedHtml.replace(fallbackQuoteRegex, sourceLinksHtml);
}

// Transform learning roadmap ASCII diagram into interactive visual ladder
const roadmapLadderRegex = /<pre><code class="language-text">【阶梯 0: 内核原理】[\s\S]*?【阶梯 11: 架构演进】[\s\S]*?<\/code><\/pre>/;
const visualLadderHtml = `
<div class="roadmap-interactive">
  <div class="roadmap-header">
    <span class="roadmap-badge">12 阶认知图谱</span>
    <h3>量化回测工业级工程梯度演进路径 (点击卡片直达对应章节)</h3>
  </div>
  <div class="ladder-grid">
    <a href="#section-1" class="ladder-card ladder-card-0">
      <div class="ladder-card-top">
        <span class="ladder-step">阶梯 0 · 内核原理</span>
        <span class="ladder-jump-tag">直达第 1 章 →</span>
      </div>
      <div class="ladder-title">1. 量化回测第一性原理与 30 行最小状态机原型</div>
      <div class="ladder-tags"><span>Cash</span><span>Shares</span><span>NAV</span><span>6大陷阱</span></div>
    </a>
    <div class="ladder-arrow">↓</div>
    <a href="#section-2" class="ladder-card ladder-card-1">
      <div class="ladder-card-top">
        <span class="ladder-step">阶梯 1 · 物理规则</span>
        <span class="ladder-jump-tag">直达第 2 章 →</span>
      </div>
      <div class="ladder-title">2. 真实世界的物理约束与微观摩擦建模</div>
      <div class="ladder-tags"><span>100股整手</span><span>T+1</span><span>万三佣金保底5元</span><span>印花税</span><span>0.1%滑点</span></div>
    </a>
    <div class="ladder-arrow">↓</div>
    <a href="#section-3" class="ladder-card ladder-card-2">
      <div class="ladder-card-top">
        <span class="ladder-step">阶梯 2 · 因果律</span>
        <span class="ladder-jump-tag">直达第 3 章 →</span>
      </div>
      <div class="ladder-title">3. 时序因果律与防偷价铁律 (Anti-Lookahead)</div>
      <div class="ladder-tags"><span>Point-in-Time</span><span>shift(1)</span><span>hist_df 解耦目录</span></div>
    </a>
    <div class="ladder-arrow">↓</div>
    <a href="#section-4" class="ladder-card ladder-card-3">
      <div class="ladder-card-top">
        <span class="ladder-step">阶梯 3 · 信号生成</span>
        <span class="ladder-jump-tag">直达第 4 章 →</span>
      </div>
      <div class="ladder-title">4. 从 AI 预测到量化交易信号 (Prediction to Signal)</div>
      <div class="ladder-tags"><span>蒙特卡洛</span><span>反转感知去偏</span><span>根号时间</span><span>绝对 Z-Score</span><span>粒子滤波</span></div>
    </a>
    <div class="ladder-arrow">↓</div>
    <a href="#section-5" class="ladder-card ladder-card-4">
      <div class="ladder-card-top">
        <span class="ladder-step">阶梯 4 · 门禁过滤</span>
        <span class="ladder-jump-tag">直达第 5 章 →</span>
      </div>
      <div class="ladder-title">5. 守门人机制：九重入场否决矩阵 (9-Stage Risk Veto)</div>
      <div class="ladder-tags"><span>绝对信噪比门槛</span><span>均线趋势确认</span><span>veto_reason</span><span>signal_df</span></div>
    </a>
    <div class="ladder-arrow">↓</div>
    <a href="#section-6" class="ladder-card ladder-card-5">
      <div class="ladder-card-top">
        <span class="ladder-step">阶梯 5 · 头寸管理</span>
        <span class="ladder-jump-tag">直达第 6 章 →</span>
      </div>
      <div class="ladder-title">6. 头寸管理：机构级风险预算定仓 (Risk Budgeting)</div>
      <div class="ladder-tags"><span>Risk Budget 1%</span><span>动态 ATR 止损</span><span>一手保底</span><span>连亏降仓防守</span></div>
    </a>
    <div class="ladder-arrow">↓</div>
    <a href="#section-7" class="ladder-card ladder-card-6">
      <div class="ladder-card-top">
        <span class="ladder-step">阶梯 6 · 出场防守</span>
        <span class="ladder-jump-tag">直达第 7 章 →</span>
      </div>
      <div class="ladder-title">7. 持仓管理：离散状态机多维离场体系 (Exit State Machine)</div>
      <div class="ladder-tags"><span>盘中止损</span><span>移动止盈</span><span>动量跃迁</span><span>backtest_df</span></div>
    </a>
    <div class="ladder-arrow">↓</div>
    <a href="#section-8" class="ladder-card ladder-card-7">
      <div class="ladder-card-top">
        <span class="ladder-step">阶梯 7 · 数据闭环</span>
        <span class="ladder-jump-tag">直达第 8 章 →</span>
      </div>
      <div class="ladder-title">8. 端到端全链路数据流实战闭环演练 (以 300418 为例)</div>
      <div class="ladder-tags"><span>300418 创业板</span><span>Stage 0~8 穿透推导</span><span>trades 台账</span></div>
    </a>
    <div class="ladder-arrow">↓</div>
    <a href="#section-9" class="ladder-card ladder-card-8">
      <div class="ladder-card-top">
        <span class="ladder-step">阶梯 8 · 成败裁判</span>
        <span class="ladder-jump-tag">直达第 9 章 →</span>
      </div>
      <div class="ladder-title">9. 成败裁判：双重评价体系 (前置 AI 体检 12 项 + 后置严密绩效 22 项)</div>
      <div class="ladder-tags"><span>12项AI体检</span><span>22项量化绩效</span><span>MFE/MAE</span><span>在市暴露</span><span>日频夏普</span></div>
    </a>
    <div class="ladder-arrow">↓</div>
    <a href="#section-10" class="ladder-card ladder-card-9">
      <div class="ladder-card-top">
        <span class="ladder-step">阶梯 9 · 工程底座</span>
        <span class="ladder-jump-tag">直达第 10 章 →</span>
      </div>
      <div class="ladder-title">10. 工业化推演体系与交付生态 (Engineering & Ecosystem)</div>
      <div class="ladder-tags"><span>Walk-Forward</span><span>指纹缓存</span><span>CLI 指令全集</span><span>Streamlit 双看板</span><span>交互图表</span></div>
    </a>
    <div class="ladder-arrow">↓</div>
    <a href="#section-11" class="ladder-card ladder-card-10">
      <div class="ladder-card-top">
        <span class="ladder-step">阶梯 10 · 变量总表</span>
        <span class="ladder-jump-tag">直达第 11 章 →</span>
      </div>
      <div class="ladder-title">11. 附录：38 项控制参数与 34 项指标数据流向速查总表</div>
      <div class="ladder-tags"><span>38项超参全集</span><span>34项指标速查</span><span>端到端数据流向</span></div>
    </a>
    <div class="ladder-arrow">↓</div>
    <a href="#section-12" class="ladder-card ladder-card-11">
      <div class="ladder-card-top">
        <span class="ladder-step">阶梯 11 · 架构演进</span>
        <span class="ladder-jump-tag">直达第 12 章 →</span>
      </div>
      <div class="ladder-title">12. 港美股多周期与生产级模块化架构演进手册 (V2.0 工业化重构)</div>
      <div class="ladder-tags"><span>分层解耦蓝图</span><span>BaseStrategy</span><span>SimulatedBroker</span><span>15m~1d多周期</span></div>
    </a>
  </div>
</div>
`;
if (roadmapLadderRegex.test(processedHtml)) {
  processedHtml = processedHtml.replace(roadmapLadderRegex, visualLadderHtml);
} else {
  console.warn('Roadmap ladder regex did not match, fallback to broader pattern');
  const fallbackRoadmapRegex = /<pre><code class="language-text">【阶梯 0: 内核原理】[\s\S]*?<\/code><\/pre>/;
  processedHtml = processedHtml.replace(fallbackRoadmapRegex, visualLadderHtml);
}

// 3. Transform Mermaid blocks
processedHtml = processedHtml.replace(
  /<pre><code class="language-mermaid">([\s\S]*?)<\/code><\/pre>/g,
  (match, p1) => {
    const rawCode = decodeEntities(p1.trim());
    return `
<div class="mermaid-block-card">
  <div class="mermaid-header">
    <div class="mermaid-title-wrap">
      <span class="mermaid-icon">🔀</span>
      <span class="mermaid-title">架构逻辑拓扑图 (Mermaid 动态渲染)</span>
    </div>
    <div class="mermaid-actions">
      <button class="action-btn" onclick="toggleMermaidSource(this)">查看源码</button>
    </div>
  </div>
  <div class="mermaid-viewport">
    <div class="mermaid">${rawCode}</div>
  </div>
  <div class="mermaid-source-code" style="display:none;">
    <pre><code>${p1.trim()}</code></pre>
  </div>
</div>`;
  }
);

// 4. Transform other Code blocks (python, bash, text, json) with SYNTAX COLOR HIGHLIGHTING & LINE NUMBERS
processedHtml = processedHtml.replace(
  /<pre><code class="language-([a-zA-Z0-9_-]+)">([\s\S]*?)<\/code><\/pre>/g,
  (match, lang, code) => {
    if (lang === 'mermaid') return match;
    const rawCode = decodeEntities(code.trim());
    return formatCodeBlock(rawCode, lang);
  }
);

// Also handle code blocks without language
processedHtml = processedHtml.replace(
  /<pre><code>([\s\S]*?)<\/code><\/pre>/g,
  (match, code) => {
    const rawCode = decodeEntities(code.trim());
    return formatCodeBlock(rawCode, 'text');
  }
);

// 5. Wrap tables in responsive containers with badges & search tools
let tableCount = 0;
processedHtml = processedHtml.replace(/<table>([\s\S]*?)<\/table>/g, (match, inner) => {
  tableCount++;
  // Smart badge replacements inside tables
  let styledInner = inner
    .replace(/>\s*(通过|PASS|卓越|优良|良好|高置信|高精拟合|显著|极显著|边界有效|可信|极度可信)\s*</gi, (m, val) => `><span class="badge badge-success">${val}</span><`)
    .replace(/>\s*(否决|FAIL|不合格|严重背离|跟踪失效|重度退化|失效)\s*</gi, (m, val) => `><span class="badge badge-danger">${val}</span><`)
    .replace(/>\s*(边缘|及格|告警|合理区间|可接受|警示)\s*</gi, (m, val) => `><span class="badge badge-warning">${val}</span><`)
    .replace(/>\s*(Stage\s*\d+(?:-\d+)?)\s*</gi, (m, val) => `><span class="badge badge-primary">${val}</span><`)
    .replace(/>\s*([+]\d+(?:\.\d+)?%?)\s*</g, (m, val) => `><span class="num-positive">${val}</span><`)
    .replace(/>\s*([−-]\d+(?:\.\d+)?%?)\s*</g, (m, val) => `><span class="num-negative">${val}</span><`);

  return `
<div class="table-card" id="table-${tableCount}">
  <div class="table-toolbar">
    <div class="table-tag">
      <span class="table-icon">📋</span>
      <span class="table-title">数据字典与参数映射表 #${tableCount}</span>
    </div>
    <div class="table-search-box">
      <input type="text" class="table-filter-input" placeholder="在表格内极速过滤关键字..." oninput="filterTable(this)">
    </div>
  </div>
  <div class="table-responsive">
    <table>${styledInner}</table>
  </div>
</div>`;
});

// 6. Extract Headings and build Hierarchical TOC
const headings = [];
let hIndex = 0;

processedHtml = processedHtml.replace(/<h([1-4])>(.*?)<\/h\1>/g, (match, level, text) => {
  hIndex++;
  const rawText = text.replace(/<[^>]+>/g, '').trim();
  
  // Make an id
  let anchorId = '';
  // Check if matches section number like "1. 量化回测..."
  const numMatch = rawText.match(/^(\d+(?:\.\d+)*(?:\.\d+)?)\s*(.*)/);
  if (numMatch) {
    anchorId = 'section-' + numMatch[1].replace(/\./g, '-');
  } else if (rawText.includes('学习路线图与目录')) {
    anchorId = 'roadmap';
  } else if (rawText.includes('详细章节目录')) {
    anchorId = 'toc-detailed';
  } else if (rawText.includes('全局假定与初始参数配置')) {
    anchorId = 'stage-0';
  } else if (rawText.includes('步骤 1：')) {
    anchorId = 'stage-1';
  } else if (rawText.includes('步骤 2：')) {
    anchorId = 'stage-2';
  } else if (rawText.includes('步骤 3：')) {
    anchorId = 'stage-3';
  } else if (rawText.includes('步骤 4：')) {
    anchorId = 'stage-4';
  } else if (rawText.includes('步骤 5：')) {
    anchorId = 'stage-5';
  } else if (rawText.includes('步骤 6：')) {
    anchorId = 'stage-6';
  } else if (rawText.includes('步骤 7：')) {
    anchorId = 'stage-7';
  } else if (rawText.includes('步骤 8：')) {
    anchorId = 'stage-8';
  } else {
    anchorId = 'heading-' + hIndex;
  }

  headings.push({
    level: parseInt(level),
    id: anchorId,
    title: rawText,
    rawText: rawText
  });

  return `<h${level} id="${anchorId}" class="doc-heading doc-heading-${level}">
    <span class="heading-anchor" onclick="copyHeadingLink('${anchorId}')" title="点击复制直达链接">#</span>
    <span class="heading-text">${text}</span>
  </h${level}>`;
});

console.log('Total processed headings:', headings.length);

// Build Sidebar HTML
function buildSidebar(headings) {
  let html = '<div class="sidebar-nav-tree">\n';
  let currentGroup = null;

  headings.forEach((h) => {
    if (h.level === 1) return; // Skip main doc title

    if (h.level === 2) {
      if (currentGroup) {
        html += '  </div></div>\n'; // close previous section
      }
      // Check if numbered chapter e.g. "1. 量化回测..."
      const match = h.title.match(/^(\d+)\.\s*(.*)/);
      const isRoadmap = h.title.includes('学习路线图');
      const badge = match ? `第 ${match[1]} 章` : (isRoadmap ? '导读' : '章节');
      const cleanTitle = match ? match[2] : h.title;

      html += `  <div class="nav-chapter-group" data-chapter="${h.id}">\n`;
      html += `    <div class="nav-chapter-header" onclick="toggleChapter(this)">\n`;
      html += `      <span class="chapter-badge">${badge}</span>\n`;
      html += `      <a href="#${h.id}" class="chapter-link" onclick="handleNavClick(event, '${h.id}')">${cleanTitle}</a>\n`;
      html += `      <span class="chapter-arrow">▾</span>\n`;
      html += `    </div>\n`;
      html += `    <div class="nav-sub-list">\n`;
      currentGroup = h.id;
    } else if (h.level === 3 && currentGroup) {
      html += `      <a href="#${h.id}" class="nav-sub-link" data-id="${h.id}" onclick="handleNavClick(event, '${h.id}')">\n`;
      html += `        <span class="sub-dot"></span>\n`;
      html += `        <span class="sub-title">${h.title}</span>\n`;
      html += `      </a>\n`;
    }
  });

  if (currentGroup) {
    html += '  </div></div>\n';
  }
  html += '</div>\n';
  return html;
}

const sidebarNavHtml = buildSidebar(headings);

// Read old HTML for exact CSS styles or use the complete design system
const oldHtml = fs.readFileSync(OUT_PATH, 'utf-8');
const styleMatch = oldHtml.match(/<style>([\s\S]*?)<\/style>/);
const cssContent = styleMatch ? styleMatch[1] : '';

// Build Full HTML Document
const fullHtml = `<!DOCTYPE html>
<html lang="zh-CN" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Kronos-Pro 量化回测系统完全解析与认知实战手册</title>
  <meta name="description" content="Kronos-Pro 工业级量化回测系统全景完全解析与实战手册：涵盖第一性原理、微观物理摩擦、因果防偷价、AI预测解耦、九重入场否决矩阵、机构风险预算定仓、离散状态机出场与 300418 闭环数值穿透。">
  <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>⚡</text></svg>">
  
  <!-- Mermaid.js for Diagrams -->
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>

  <style>
${cssContent}
  </style>
</head>
<body>

  <!-- Top Sticky Header -->
  <header class="top-navbar">
    <div class="brand-section">
      <button class="tool-icon-btn mobile-menu-btn" onclick="toggleMobileSidebar()" aria-label="Toggle Menu">☰</button>
      <div class="brand-icon">⚡</div>
      <div class="brand-text-wrap">
        <span class="brand-title">
          Kronos-Pro
          <span class="version-tag">v2.0 工业级</span>
        </span>
        <span class="brand-sub">量化回测系统完全解析与认知实战手册</span>
      </div>
    </div>

    <div class="nav-tools">
      <button class="search-trigger-btn" onclick="openSearchModal()" title="快捷键 Cmd/Ctrl + K">
        <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
        <span>全局极速搜索...</span>
        <span class="search-kbd">⌘K</span>
      </button>

      <button class="tool-icon-btn" id="theme-toggle-btn" onclick="toggleTheme()" title="切换明暗主题">
        🌙
      </button>

      <button class="tool-icon-btn" onclick="window.print()" title="打印或导出为 PDF">
        🖨️
      </button>
    </div>

    <!-- Reading Progress Bar -->
    <div class="reading-progress-track">
      <div class="reading-progress-bar" id="reading-progress-bar"></div>
    </div>
  </header>

  <!-- Layout Container -->
  <div class="doc-layout">
    
    <!-- Left Sidebar -->
    <aside class="doc-sidebar" id="doc-sidebar">
      <div class="sidebar-search-box">
        <input type="text" class="sidebar-filter-input" id="sidebar-filter-input" placeholder="在目录树中过滤章节..." oninput="filterSidebar(this.value)">
      </div>

      <div class="quick-jump-container">
        <a href="#section-1" class="quick-jump-pill">0 原理</a>
        <a href="#section-2" class="quick-jump-pill">1 物理</a>
        <a href="#section-3" class="quick-jump-pill">2 因果</a>
        <a href="#section-4" class="quick-jump-pill">3 信号</a>
        <a href="#section-5" class="quick-jump-pill">4 门禁</a>
        <a href="#section-6" class="quick-jump-pill">5 定仓</a>
        <a href="#section-7" class="quick-jump-pill">6 状态机</a>
        <a href="#section-8" class="quick-jump-pill">7 闭环</a>
        <a href="#section-9" class="quick-jump-pill">8 评价</a>
        <a href="#section-10" class="quick-jump-pill">9 工程</a>
        <a href="#section-11" class="quick-jump-pill">10 字典</a>
        <a href="#section-12" class="quick-jump-pill">11 架构</a>
      </div>

      ${sidebarNavHtml}
    </aside>

    <!-- Main Content -->
    <main class="doc-main">
      <article class="doc-article" id="doc-article">
        
        <!-- Hero Header -->
        <div class="doc-hero">
          <div class="hero-meta-tags">
            <span class="meta-tag">Python 3.12+</span>
            <span class="meta-tag purple">12 大认知阶梯</span>
            <span class="meta-tag green">38 项核心超参</span>
            <span class="meta-tag">34 项严密指标</span>
            <span class="meta-tag red">22 项量化绩效体检</span>
            <span class="meta-tag purple">V2.0 生产级模块化解耦</span>
          </div>
          <h1>Kronos-Pro 量化回测系统完全解析与认知实战手册</h1>
          <p class="doc-hero-desc">
            从第一性原理、微观物理摩擦、因果时序律，到 AI 预测特征解耦、九重风险否决、离散状态机出场、300418 端到端实战数值穿透，以及港美股多周期与生产级模块化解耦架构演进。专为量化研发、高频回测与工业化实盘部署设计。
          </p>
        </div>

        <!-- Rendered Body Content -->
        <div class="markdown-body">
          ${processedHtml}
        </div>

      </article>
    </main>

  </div>

  <!-- Floating Utilities -->
  <div class="floating-tools">
    <button class="float-btn" onclick="scrollToTop()" title="返回顶部">↑</button>
  </div>

  <!-- Search Modal Overlay -->
  <div class="search-modal-overlay" id="search-modal-overlay" onclick="handleModalOverlayClick(event)">
    <div class="search-modal">
      <div class="search-input-wrap">
        <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
        <input type="text" class="modal-search-input" id="modal-search-input" placeholder="输入关键字搜索章节、公式、参数或代码..." oninput="performModalSearch(this.value)">
        <span class="modal-close-kbd" onclick="closeSearchModal()">ESC</span>
      </div>
      <div class="search-results-list" id="search-results-list">
        <div style="padding: 20px; text-align: center; color: var(--text-dim); font-size: 0.85rem;">
          输入关键字立即在整篇手册中实时全文检索...
        </div>
      </div>
    </div>
  </div>

  <!-- Toast Notification -->
  <div class="toast-msg" id="toast-msg">已复制到剪贴板</div>

  <!-- Client-side Interactive Scripts -->
  <script>
    // Initialize Mermaid
    document.addEventListener("DOMContentLoaded", function() {
      if (typeof mermaid !== 'undefined') {
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        mermaid.initialize({
          startOnLoad: true,
          theme: isDark ? 'dark' : 'default',
          securityLevel: 'loose',
          flowchart: {
            useMaxWidth: true,
            htmlLabels: true,
            curve: 'basis'
          }
        });
      }
    });

    // Theme Toggle
    function toggleTheme() {
      const html = document.documentElement;
      const current = html.getAttribute('data-theme');
      const next = current === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-theme', next);
      localStorage.setItem('kronos_doc_theme', next);
      
      const btn = document.getElementById('theme-toggle-btn');
      btn.innerText = next === 'dark' ? '🌙' : '☀️';

      if (typeof mermaid !== 'undefined') {
        mermaid.initialize({
          startOnLoad: false,
          theme: next === 'dark' ? 'dark' : 'default'
        });
      }
    }

    // Load Saved Theme
    const savedTheme = localStorage.getItem('kronos_doc_theme');
    if (savedTheme) {
      document.documentElement.setAttribute('data-theme', savedTheme);
      const btn = document.getElementById('theme-toggle-btn');
      if (btn) btn.innerText = savedTheme === 'dark' ? '🌙' : '☀️';
    }

    // Mobile Sidebar Toggle
    function toggleMobileSidebar() {
      const sidebar = document.getElementById('doc-sidebar');
      sidebar.classList.toggle('open');
    }

    // Scroll to Top
    function scrollToTop() {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // Copy Toast Helper
    function showToast(msg) {
      const toast = document.getElementById('toast-msg');
      toast.innerText = msg;
      toast.classList.add('show');
      setTimeout(() => toast.classList.remove('show'), 2000);
    }

    // Copy Code Block (Pristine pure code copying without line numbers)
    function copyCodeBlock(btn) {
      const card = btn.closest('.code-card');
      const rawEncoded = card.getAttribute('data-raw');
      let textToCopy = '';
      if (rawEncoded) {
        textToCopy = decodeURIComponent(rawEncoded);
      } else {
        const lines = card.querySelectorAll('.line-text');
        textToCopy = Array.from(lines).map(l => l.innerText).join('\\n');
      }
      navigator.clipboard.writeText(textToCopy).then(() => {
        btn.classList.add('copied');
        btn.querySelector('span').innerText = '已复制!';
        showToast('代码已成功复制');
        setTimeout(() => {
          btn.classList.remove('copied');
          btn.querySelector('span').innerText = '复制';
        }, 2000);
      });
    }

    // Copy Heading Anchor Link
    function copyHeadingLink(anchorId) {
      const url = window.location.origin + window.location.pathname + '#' + anchorId;
      navigator.clipboard.writeText(url).then(() => {
        showToast('已复制小节直达链接');
      });
    }

    // Toggle Mermaid Source Code
    function toggleMermaidSource(btn) {
      const card = btn.closest('.mermaid-block-card');
      const src = card.querySelector('.mermaid-source-code');
      const isHidden = src.style.display === 'none';
      src.style.display = isHidden ? 'block' : 'none';
      btn.innerText = isHidden ? '收起源码' : '查看源码';
    }

    // Table Filter
    function filterTable(input) {
      const query = input.value.toLowerCase().trim();
      const table = input.closest('.table-card').querySelector('table');
      const rows = table.querySelectorAll('tbody tr');

      rows.forEach(row => {
        const text = row.innerText.toLowerCase();
        if (!query || text.includes(query)) {
          row.style.display = '';
        } else {
          row.style.display = 'none';
        }
      });
    }

    // Sidebar Chapter Accordion Toggle
    function toggleChapter(header) {
      const group = header.closest('.nav-chapter-group');
      group.classList.toggle('collapsed');
    }

    // Sidebar Filter
    function filterSidebar(query) {
      const val = query.toLowerCase().trim();
      const groups = document.querySelectorAll('.nav-chapter-group');

      groups.forEach(group => {
        const subLinks = group.querySelectorAll('.nav-sub-link');
        let anyVisible = false;

        subLinks.forEach(link => {
          const text = link.innerText.toLowerCase();
          if (!val || text.includes(val)) {
            link.style.display = 'flex';
            anyVisible = true;
          } else {
            link.style.display = 'none';
          }
        });

        const headerText = group.querySelector('.nav-chapter-header').innerText.toLowerCase();
        if (headerText.includes(val) || anyVisible) {
          group.style.display = 'block';
          if (val) group.classList.remove('collapsed');
        } else {
          group.style.display = 'none';
        }
      });
    }

    // Scroll Progress & Active Chapter Tracking
    window.addEventListener('scroll', () => {
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      const scrolled = window.scrollY;
      const progress = docHeight > 0 ? (scrolled / docHeight) * 100 : 0;
      
      const bar = document.getElementById('reading-progress-bar');
      if (bar) bar.style.width = progress + '%';

      // Highlight active TOC item
      const headings = document.querySelectorAll('.doc-heading');
      let currentId = '';

      headings.forEach(h => {
        const rect = h.getBoundingClientRect();
        if (rect.top <= 140) {
          currentId = h.id;
        }
      });

      if (currentId) {
        document.querySelectorAll('.nav-sub-link').forEach(link => {
          if (link.getAttribute('data-id') === currentId) {
            link.classList.add('active');
            const parentGroup = link.closest('.nav-chapter-group');
            if (parentGroup) parentGroup.classList.remove('collapsed');
          } else {
            link.classList.remove('active');
          }
        });
      }
    });

    // Handle Nav Click smoothly
    function handleNavClick(e, id) {
      const target = document.getElementById(id);
      if (target) {
        document.getElementById('doc-sidebar').classList.remove('open');
      }
    }

    // Modal Search System
    const searchModalOverlay = document.getElementById('search-modal-overlay');
    const modalSearchInput = document.getElementById('modal-search-input');
    const searchResultsList = document.getElementById('search-results-list');

    // Build Search Index from Article
    let searchIndex = [];
    document.addEventListener('DOMContentLoaded', () => {
      const headings = document.querySelectorAll('.doc-heading');
      headings.forEach(h => {
        let snippet = '';
        let sibling = h.nextElementSibling;
        while (sibling && !sibling.classList.contains('doc-heading')) {
          snippet += ' ' + sibling.innerText;
          sibling = sibling.nextElementSibling;
          if (snippet.length > 250) break;
        }
        searchIndex.push({
          id: h.id,
          title: h.innerText.replace('#', '').trim(),
          level: h.tagName,
          snippet: snippet.trim()
        });
      });
    });

    function openSearchModal() {
      searchModalOverlay.style.display = 'flex';
      modalSearchInput.focus();
      modalSearchInput.select();
    }

    function closeSearchModal() {
      searchModalOverlay.style.display = 'none';
      modalSearchInput.value = '';
    }

    function handleModalOverlayClick(e) {
      if (e.target === searchModalOverlay) {
        closeSearchModal();
      }
    }

    document.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        openSearchModal();
      } else if (e.key === 'Escape') {
        closeSearchModal();
      }
    });

    function performModalSearch(query) {
      const q = query.toLowerCase().trim();
      if (!q) {
        searchResultsList.innerHTML = '<div style="padding: 20px; text-align: center; color: var(--text-dim); font-size: 0.85rem;">输入关键字立即在整篇手册中实时全文检索...</div>';
        return;
      }

      const matches = searchIndex.filter(item => 
        item.title.toLowerCase().includes(q) || item.snippet.toLowerCase().includes(q)
      ).slice(0, 15);

      if (matches.length === 0) {
        searchResultsList.innerHTML = '<div style="padding: 24px; text-align: center; color: var(--text-muted); font-size: 0.9rem;">未找到与 "<strong>' + query + '</strong>" 相关的章节内容</div>';
        return;
      }

      let html = '';
      matches.forEach(m => {
        const hlTitle = m.title.replace(new RegExp(q, 'gi'), match => '<span style="color:var(--color-primary);font-weight:700;">' + match + '</span>');
        html += '<a href="#' + m.id + '" class="search-result-item" onclick="closeSearchModal()">' +
                '  <span class="result-title">' + hlTitle + '</span>' +
                '  <span class="result-snippet">' + m.snippet.slice(0, 120) + '...</span>' +
                '</a>';
      });

      searchResultsList.innerHTML = html;
    }
  </script>
</body>
</html>
`;

console.log('Writing output file to:', OUT_PATH);
fs.writeFileSync(OUT_PATH, fullHtml, 'utf-8');
console.log('Done! Generated single-page HTML successfully. File size:', (fullHtml.length / 1024).toFixed(1), 'KB');
