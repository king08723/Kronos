const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

let hljs;
try {
  hljs = require('/tmp/hljs_tool/node_modules/highlight.js');
} catch (e) {
  try {
    hljs = require('highlight.js');
  } catch (e2) {
    console.warn('highlight.js not found, continuing with basic escaping');
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
      .replace(/\b([0-9]{6}|[0-9]{5}|AAPL)\b/g, '<span class="cmd-arg">$1</span>')
      .replace(/(backtest\/[a-zA-Z0-9_]+\.(?:py|csv|html|pdf))/g, '<span class="cmd-path">$1</span>');
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
    l = l.replace(/\b(sqrt|mean|std|clip|max|min|count|sum|valid|sign|abs|rolling)\b/g, '<span class="calc-func">$1</span>');
    // Key indicators & variables
    l = l.replace(/\b(vol_14d|volatility_scaled|atr|atr_pct|close_prev|expected_return|expected_return_adj|signal_score|risk_budget|stop_distance|stop_price|shares_to_buy|buy_price|sell_price|portfolio_value|cash|total_return|sharpe|ESS|weights|pf_confidence|target_horizon|optimal_horizon|pred_close|pred_path_max|pred_min_down|pred_slope|position_scale)\b/g, '<span class="calc-var">$1</span>');
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
    } else if (rawCode.includes('=') && (rawCode.includes('sqrt') || rawCode.includes('std') || rawCode.includes('atr') || rawCode.includes('portfolio_value') || rawCode.includes('mean'))) {
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
const sourceQuoteRegex = /<blockquote>\s*<p>对应源码：[\s\S]*?深度归因研报导出：[\s\S]*?<\/p>\s*<\/blockquote>/;
const sourceLinksHtml = `
<div class="source-modules-grid">
  <div class="module-card">
    <div class="module-icon">⚡</div>
    <div class="module-info">
      <span class="module-title">核心回测引擎</span>
      <a class="module-link" href="file:///Users/jyy/Documents/github_projects/Kronos/backtest/run_backtest_kronos_pro.py">run_backtest_kronos_pro.py</a>
      <span class="module-desc">4000+行全链路回测、九重风控与离散状态机</span>
    </div>
  </div>
  <div class="module-card">
    <div class="module-icon">🏛️</div>
    <div class="module-info">
      <span class="module-title">多市场物理画像</span>
      <a class="module-link" href="file:///Users/jyy/Documents/github_projects/Kronos/backtest/market_rules.py">market_rules.py</a>
      <span class="module-desc">A股/港股/美股整手、税费、滑点与涨跌停约束</span>
    </div>
  </div>
  <div class="module-card">
    <div class="module-icon">📦</div>
    <div class="module-info">
      <span class="module-title">批量推演调度</span>
      <a class="module-link" href="file:///Users/jyy/Documents/github_projects/Kronos/backtest/run_backtest_batch.py">run_backtest_batch.py</a>
      <span class="module-desc">股票池批量调度、指纹缓存与胜率矩阵统计</span>
    </div>
  </div>
  <div class="module-card">
    <div class="module-icon">📊</div>
    <div class="module-info">
      <span class="module-title">统一量化中枢看板</span>
      <a class="module-link" href="file:///Users/jyy/Documents/github_projects/Kronos/backtest/app_backtest_dashboard.py">app_backtest_dashboard.py</a>
      <span class="module-desc">Streamlit 交互式多维光标吸附与参数热重载</span>
    </div>
  </div>
  <div class="module-card">
    <div class="module-icon">📑</div>
    <div class="module-info">
      <span class="module-title">深度归因研报导出</span>
      <a class="module-link" href="file:///Users/jyy/Documents/github_projects/Kronos/backtest/generate_strategy_report_pdf.py">generate_strategy_report_pdf.py</a>
      <span class="module-desc">A4 标准化研报 HTML 与专业 PDF 自动生成</span>
    </div>
  </div>
</div>
`;
processedHtml = processedHtml.replace(sourceQuoteRegex, sourceLinksHtml);

// Transform learning roadmap ASCII diagram into interactive visual ladder
const roadmapLadderRegex = /<pre><code class="language-text">【阶梯 0: 内核原理】[\s\S]*?【阶梯 10: 变量总表】[\s\S]*?<\/code><\/pre>/;
const visualLadderHtml = `
<div class="roadmap-interactive">
  <div class="roadmap-header">
    <span class="roadmap-badge">11 阶认知图谱</span>
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
      <div class="ladder-tags"><span>100股整手</span><span>T+1</span><span>万三佣金</span><span>印花税</span><span>0.1%滑点</span></div>
    </a>
    <div class="ladder-arrow">↓</div>
    <a href="#section-3" class="ladder-card ladder-card-2">
      <div class="ladder-card-top">
        <span class="ladder-step">阶梯 2 · 因果律</span>
        <span class="ladder-jump-tag">直达第 3 章 →</span>
      </div>
      <div class="ladder-title">3. 时序因果律与防偷价铁律 (Anti-Lookahead)</div>
      <div class="ladder-tags"><span>Point-in-Time</span><span>shift(1)</span><span>hist_df</span></div>
    </a>
    <div class="ladder-arrow">↓</div>
    <a href="#section-4" class="ladder-card ladder-card-3">
      <div class="ladder-card-top">
        <span class="ladder-step">阶梯 3 · 信号生成</span>
        <span class="ladder-jump-tag">直达第 4 章 →</span>
      </div>
      <div class="ladder-title">4. 从 AI 预测到量化交易信号 (Prediction to Signal)</div>
      <div class="ladder-tags"><span>蒙特卡洛</span><span>去偏(Debias)</span><span>根号时间</span><span>Z-Score</span><span>粒子滤波</span></div>
    </a>
    <div class="ladder-arrow">↓</div>
    <a href="#section-5" class="ladder-card ladder-card-4">
      <div class="ladder-card-top">
        <span class="ladder-step">阶梯 4 · 门禁过滤</span>
        <span class="ladder-jump-tag">直达第 5 章 →</span>
      </div>
      <div class="ladder-title">5. 守门人机制：九重入场否决矩阵 (9-Stage Risk Veto)</div>
      <div class="ladder-tags"><span>9-Stage Veto</span><span>均线确认</span><span>veto_reason</span><span>signal_df</span></div>
    </a>
    <div class="ladder-arrow">↓</div>
    <a href="#section-6" class="ladder-card ladder-card-5">
      <div class="ladder-card-top">
        <span class="ladder-step">阶梯 5 · 头寸管理</span>
        <span class="ladder-jump-tag">直达第 6 章 →</span>
      </div>
      <div class="ladder-title">6. 头寸管理：机构级风险预算定仓 (Risk Budgeting)</div>
      <div class="ladder-tags"><span>Risk Budget 1%</span><span>动态 ATR</span><span>一手保底</span><span>连亏降仓</span></div>
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
      <div class="ladder-tags"><span>300418 实例</span><span>Stage 0~8 穿透推导</span><span>trades 台账</span></div>
    </a>
    <div class="ladder-arrow">↓</div>
    <a href="#section-9" class="ladder-card ladder-card-8">
      <div class="ladder-card-top">
        <span class="ladder-step">阶梯 8 · 成败裁判</span>
        <span class="ladder-jump-tag">直达第 9 章 →</span>
      </div>
      <div class="ladder-title">9. 成败裁判：双重评价体系 (前置体检 + 绩效修正)</div>
      <div class="ladder-tags"><span>12项AI体检</span><span>17项严密绩效</span><span>日频夏普</span><span>现金盈亏比</span></div>
    </a>
    <div class="ladder-arrow">↓</div>
    <a href="#section-10" class="ladder-card ladder-card-9">
      <div class="ladder-card-top">
        <span class="ladder-step">阶梯 9 · 工程底座</span>
        <span class="ladder-jump-tag">直达第 10 章 →</span>
      </div>
      <div class="ladder-title">10. 工业化推演体系与交付生态 (Engineering & Ecosystem)</div>
      <div class="ladder-tags"><span>Walk-Forward</span><span>指纹缓存</span><span>CLI 指令</span><span>Streamlit 看板</span></div>
    </a>
    <div class="ladder-arrow">↓</div>
    <a href="#section-11" class="ladder-card ladder-card-10">
      <div class="ladder-card-top">
        <span class="ladder-step">阶梯 10 · 变量总表</span>
        <span class="ladder-jump-tag">直达第 11 章 →</span>
      </div>
      <div class="ladder-title">11. 附录：31 项控制参数与 29 项指标数据流向速查总表</div>
      <div class="ladder-tags"><span>超参全集</span><span>指标速查字典</span><span>端到端生命周期</span></div>
    </a>
  </div>
</div>
`;
processedHtml = processedHtml.replace(roadmapLadderRegex, visualLadderHtml);

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
    .replace(/>\s*(边缘|及格|告警|合理区间|可接受)\s*</gi, (m, val) => `><span class="badge badge-warning">${val}</span><`)
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
    /* CSS Variables Design System */
    :root {
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans", sans-serif;
      --font-mono: "Fira Code", "JetBrains Mono", SFMono-Regular, Menlo, Monaco, Consolas, "Courier New", monospace;
      
      --color-primary: #38bdf8;
      --color-primary-rgb: 56, 189, 248;
      --color-primary-dark: #0284c7;
      --color-secondary: #818cf8;
      --color-accent: #a855f7;
      
      --color-success: #10b981;
      --color-warning: #f59e0b;
      --color-danger: #f43f5e;
      --color-info: #0ea5e9;

      /* Dark Mode Palette (Default) */
      --bg-body: #0b0f19;
      --bg-surface: #111827;
      --bg-surface-elevated: #1a2234;
      --bg-sidebar: #0d1322;
      --bg-card: rgba(17, 24, 39, 0.75);
      --bg-card-hover: rgba(26, 34, 52, 0.9);
      --bg-code: #0c1018;
      --bg-code-header: #141b29;
      --bg-table-header: #151d30;
      --bg-table-row-alt: rgba(255, 255, 255, 0.015);
      --bg-table-row-hover: rgba(56, 189, 248, 0.08);

      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-focus: #38bdf8;
      --border-table: rgba(255, 255, 255, 0.09);

      --text-main: #e2e8f0;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --text-heading: #f8fafc;
      --text-link: #38bdf8;

      --sidebar-width: 320px;
      --header-height: 64px;
      --content-max-width: 1040px;

      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 14px;
      --radius-xl: 18px;

      --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.25);
      --shadow-md: 0 4px 20px rgba(0, 0, 0, 0.35);
      --shadow-lg: 0 8px 30px rgba(0, 0, 0, 0.5);
      --shadow-glow: 0 0 25px rgba(56, 189, 248, 0.2);

      --line-height-body: 1.75;
      --transition-base: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }

    [data-theme="light"] {
      --bg-body: #f8fafc;
      --bg-surface: #ffffff;
      --bg-surface-elevated: #f1f5f9;
      --bg-sidebar: #ffffff;
      --bg-card: #ffffff;
      --bg-card-hover: #f8fafc;
      --bg-code: #f8fafc;
      --bg-code-header: #eef2f6;
      --bg-table-header: #f1f5f9;
      --bg-table-row-alt: #f8fafc;
      --bg-table-row-hover: rgba(2, 132, 199, 0.06);

      --border-subtle: #e2e8f0;
      --border-focus: #0284c7;
      --border-table: #e2e8f0;

      --text-main: #1e293b;
      --text-muted: #64748b;
      --text-dim: #94a3b8;
      --text-heading: #0f172a;
      --text-link: #0284c7;

      --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.05);
      --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.08);
      --shadow-lg: 0 12px 30px rgba(0, 0, 0, 0.12);
      --shadow-glow: 0 0 20px rgba(2, 132, 199, 0.15);
    }

    /* Reset & Base */
    *, *::before, *::after {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    html {
      scroll-behavior: smooth;
      font-size: 15px;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }

    body {
      font-family: var(--font-sans);
      background-color: var(--bg-body);
      color: var(--text-main);
      line-height: var(--line-height-body);
      overflow-x: hidden;
      transition: background-color 0.25s ease, color 0.25s ease;
    }

    /* Top Sticky Header */
    .top-navbar {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      height: var(--header-height);
      background: rgba(13, 19, 34, 0.88);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      border-bottom: 1px solid var(--border-subtle);
      z-index: 1000;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      transition: var(--transition-base);
    }

    [data-theme="light"] .top-navbar {
      background: rgba(255, 255, 255, 0.88);
      border-bottom-color: #e2e8f0;
    }

    .brand-section {
      display: flex;
      align-items: center;
      gap: 12px;
      text-decoration: none;
    }

    .brand-icon {
      width: 36px;
      height: 36px;
      border-radius: var(--radius-md);
      background: linear-gradient(135deg, #38bdf8, #818cf8);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.3rem;
      box-shadow: 0 0 14px rgba(56, 189, 248, 0.4);
    }

    .brand-text-wrap {
      display: flex;
      flex-direction: column;
    }

    .brand-title {
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--text-heading);
      letter-spacing: -0.01em;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .version-tag {
      font-size: 0.68rem;
      font-weight: 600;
      padding: 2px 7px;
      border-radius: 999px;
      background: rgba(56, 189, 248, 0.15);
      color: var(--color-primary);
      border: 1px solid rgba(56, 189, 248, 0.3);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .brand-sub {
      font-size: 0.72rem;
      color: var(--text-muted);
    }

    /* Top Tools Bar */
    .nav-tools {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .search-trigger-btn {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 7px 14px;
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      color: var(--text-muted);
      cursor: pointer;
      font-size: 0.85rem;
      transition: var(--transition-base);
    }

    .search-trigger-btn:hover {
      border-color: var(--color-primary);
      color: var(--text-main);
      box-shadow: var(--shadow-glow);
    }

    .search-kbd {
      padding: 2px 6px;
      background: var(--bg-body);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      font-size: 0.72rem;
      font-family: var(--font-mono);
      color: var(--text-dim);
    }

    .tool-icon-btn {
      width: 36px;
      height: 36px;
      border-radius: var(--radius-md);
      border: 1px solid var(--border-subtle);
      background: var(--bg-surface-elevated);
      color: var(--text-muted);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1rem;
      transition: var(--transition-base);
    }

    .tool-icon-btn:hover {
      color: var(--text-heading);
      border-color: var(--color-primary);
      background: var(--bg-card-hover);
    }

    /* Top Reading Progress Bar */
    .reading-progress-track {
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      height: 3px;
      background: transparent;
      overflow: hidden;
    }

    .reading-progress-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #38bdf8, #818cf8, #a855f7);
      transition: width 0.1s ease-out;
    }

    /* Layout Wrapper */
    .doc-layout {
      display: flex;
      margin-top: var(--header-height);
      min-height: calc(100vh - var(--header-height));
    }

    /* Left Sidebar */
    .doc-sidebar {
      width: var(--sidebar-width);
      position: fixed;
      top: var(--header-height);
      bottom: 0;
      left: 0;
      background: var(--bg-sidebar);
      border-right: 1px solid var(--border-subtle);
      overflow-y: auto;
      padding: 20px 16px 40px 16px;
      z-index: 900;
      display: flex;
      flex-direction: column;
      gap: 16px;
      transition: transform 0.3s ease;
    }

    .sidebar-search-box {
      position: sticky;
      top: 0;
      background: var(--bg-sidebar);
      padding-bottom: 8px;
      z-index: 10;
    }

    .sidebar-filter-input {
      width: 100%;
      padding: 9px 12px;
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      color: var(--text-main);
      font-size: 0.85rem;
      outline: none;
      transition: var(--transition-base);
    }

    .sidebar-filter-input:focus {
      border-color: var(--color-primary);
      box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15);
    }

    /* Quick Jump Pill Carousel */
    .quick-jump-container {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--border-subtle);
    }

    .quick-jump-pill {
      font-size: 0.72rem;
      padding: 3px 8px;
      border-radius: 999px;
      background: var(--bg-surface-elevated);
      color: var(--text-muted);
      text-decoration: none;
      border: 1px solid var(--border-subtle);
      transition: var(--transition-base);
    }

    .quick-jump-pill:hover {
      background: rgba(56, 189, 248, 0.12);
      color: var(--color-primary);
      border-color: rgba(56, 189, 248, 0.3);
    }

    /* Sidebar Navigation Tree */
    .sidebar-nav-tree {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .nav-chapter-group {
      border-radius: var(--radius-md);
      overflow: hidden;
      transition: var(--transition-base);
    }

    .nav-chapter-header {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border-radius: var(--radius-md);
      cursor: pointer;
      user-select: none;
      transition: var(--transition-base);
    }

    .nav-chapter-header:hover {
      background: var(--bg-surface-elevated);
    }

    .chapter-badge {
      font-size: 0.65rem;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: var(--radius-sm);
      background: rgba(56, 189, 248, 0.12);
      color: var(--color-primary);
      white-space: nowrap;
    }

    .chapter-link {
      flex: 1;
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-main);
      text-decoration: none;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .chapter-arrow {
      font-size: 0.75rem;
      color: var(--text-dim);
      transition: transform 0.2s ease;
    }

    .nav-chapter-group.collapsed .chapter-arrow {
      transform: rotate(-90deg);
    }

    .nav-chapter-group.collapsed .nav-sub-list {
      display: none;
    }

    .nav-sub-list {
      padding-left: 14px;
      margin-top: 4px;
      display: flex;
      flex-direction: column;
      gap: 2px;
      border-left: 2px solid var(--border-subtle);
      margin-left: 14px;
    }

    .nav-sub-link {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 5px 8px;
      border-radius: var(--radius-sm);
      color: var(--text-muted);
      text-decoration: none;
      font-size: 0.8rem;
      transition: var(--transition-base);
    }

    .nav-sub-link:hover {
      color: var(--text-heading);
      background: var(--bg-surface-elevated);
    }

    .nav-sub-link.active {
      color: var(--color-primary);
      font-weight: 600;
      background: rgba(56, 189, 248, 0.1);
      border-left: 2px solid var(--color-primary);
    }

    .sub-dot {
      width: 4px;
      height: 4px;
      border-radius: 50%;
      background: var(--text-dim);
      flex-shrink: 0;
    }

    .nav-sub-link.active .sub-dot {
      background: var(--color-primary);
      box-shadow: 0 0 6px var(--color-primary);
    }

    .sub-title {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    /* Main Content Area */
    .doc-main {
      margin-left: var(--sidebar-width);
      flex: 1;
      padding: 40px 48px 80px 48px;
      max-width: calc(100% - var(--sidebar-width));
      display: flex;
      justify-content: center;
    }

    .doc-article {
      width: 100%;
      max-width: var(--content-max-width);
    }

    /* Hero Section */
    .doc-hero {
      padding: 36px 40px;
      background: linear-gradient(135deg, rgba(31, 41, 55, 0.75) 0%, rgba(17, 24, 39, 0.95) 100%);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-xl);
      margin-bottom: 40px;
      position: relative;
      overflow: hidden;
      box-shadow: var(--shadow-lg);
    }

    [data-theme="light"] .doc-hero {
      background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
      border-color: #e2e8f0;
    }

    .doc-hero::after {
      content: "";
      position: absolute;
      top: -50%;
      right: -20%;
      width: 340px;
      height: 340px;
      background: radial-gradient(circle, rgba(56, 189, 248, 0.18) 0%, transparent 70%);
      pointer-events: none;
    }

    .hero-meta-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 18px;
    }

    .meta-tag {
      font-size: 0.72rem;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 999px;
      background: rgba(56, 189, 248, 0.1);
      color: var(--color-primary);
      border: 1px solid rgba(56, 189, 248, 0.25);
    }

    .meta-tag.purple {
      background: rgba(168, 85, 247, 0.1);
      color: var(--color-accent);
      border-color: rgba(168, 85, 247, 0.25);
    }

    .meta-tag.green {
      background: rgba(16, 185, 129, 0.1);
      color: var(--color-success);
      border-color: rgba(16, 185, 129, 0.25);
    }

    .doc-hero h1 {
      font-size: 2.2rem;
      font-weight: 800;
      line-height: 1.3;
      color: var(--text-heading);
      letter-spacing: -0.02em;
      margin-bottom: 16px;
      background: linear-gradient(135deg, #ffffff 40%, #94a3b8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    [data-theme="light"] .doc-hero h1 {
      background: linear-gradient(135deg, #0f172a 40%, #334155 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .doc-hero-desc {
      font-size: 1.05rem;
      color: var(--text-muted);
      line-height: 1.6;
      margin-bottom: 8px;
    }

    /* Source Modules Cards */
    .source-modules-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
      margin: 24px 0 36px 0;
    }

    .module-card {
      display: flex;
      align-items: flex-start;
      gap: 12px;
      padding: 14px 16px;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      backdrop-filter: blur(10px);
      transition: var(--transition-base);
    }

    .module-card:hover {
      border-color: var(--color-primary);
      transform: translateY(-2px);
      box-shadow: var(--shadow-md);
      background: var(--bg-card-hover);
    }

    .module-icon {
      font-size: 1.4rem;
      padding: 8px;
      border-radius: var(--radius-md);
      background: var(--bg-surface-elevated);
      line-height: 1;
    }

    .module-info {
      display: flex;
      flex-direction: column;
      gap: 3px;
    }

    .module-title {
      font-size: 0.78rem;
      font-weight: 700;
      color: var(--color-primary);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .module-link {
      font-family: var(--font-mono);
      font-size: 0.88rem;
      font-weight: 600;
      color: var(--text-heading);
      text-decoration: none;
    }

    .module-link:hover {
      color: var(--color-primary);
      text-decoration: underline;
    }

    .module-desc {
      font-size: 0.75rem;
      color: var(--text-muted);
    }

    /* Interactive Visual Roadmap */
    .roadmap-interactive {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-xl);
      padding: 24px;
      margin: 28px 0;
      box-shadow: var(--shadow-sm);
    }

    .roadmap-header {
      margin-bottom: 20px;
    }

    .roadmap-badge {
      font-size: 0.7rem;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 999px;
      background: rgba(168, 85, 247, 0.15);
      color: var(--color-accent);
      border: 1px solid rgba(168, 85, 247, 0.3);
      display: inline-block;
      margin-bottom: 6px;
    }

    .roadmap-header h3 {
      font-size: 1.15rem;
      color: var(--text-heading);
      font-weight: 700;
    }

    .ladder-grid {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .ladder-card {
      display: flex;
      flex-direction: column;
      gap: 6px;
      padding: 12px 16px;
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      text-decoration: none;
      transition: var(--transition-base);
    }

    .ladder-card:hover {
      border-color: var(--color-primary);
      background: var(--bg-card-hover);
      transform: translateX(4px);
      box-shadow: var(--shadow-sm);
    }

    .ladder-card-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .ladder-step {
      font-size: 0.72rem;
      font-weight: 700;
      color: var(--color-primary);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .ladder-jump-tag {
      font-size: 0.72rem;
      color: var(--text-dim);
      font-family: var(--font-mono);
      transition: color 0.15s ease;
    }

    .ladder-card:hover .ladder-jump-tag {
      color: var(--color-primary);
    }

    .ladder-title {
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text-heading);
    }

    .ladder-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }

    .ladder-tags span {
      font-size: 0.68rem;
      padding: 2px 6px;
      border-radius: var(--radius-sm);
      background: rgba(255, 255, 255, 0.05);
      color: var(--text-muted);
      border: 1px solid var(--border-subtle);
    }

    [data-theme="light"] .ladder-tags span {
      background: #e2e8f0;
      color: #475569;
    }

    .ladder-arrow {
      text-align: center;
      color: var(--color-primary);
      font-size: 1.1rem;
      line-height: 1;
      opacity: 0.7;
    }

    /* Headings */
    .doc-heading {
      color: var(--text-heading);
      position: relative;
      margin-top: 2.2em;
      margin-bottom: 0.8em;
      scroll-margin-top: 80px;
    }

    .doc-heading-1 {
      display: none; /* Already rendered in Hero */
    }

    .doc-heading-2 {
      font-size: 1.65rem;
      font-weight: 700;
      letter-spacing: -0.015em;
      border-bottom: 1px solid var(--border-subtle);
      padding-bottom: 0.4em;
    }

    .doc-heading-3 {
      font-size: 1.25rem;
      font-weight: 600;
      letter-spacing: -0.01em;
      margin-top: 1.8em;
    }

    .doc-heading-4 {
      font-size: 1.05rem;
      font-weight: 600;
      margin-top: 1.4em;
      color: var(--color-primary);
    }

    .heading-anchor {
      position: absolute;
      left: -1.2em;
      color: var(--text-dim);
      cursor: pointer;
      opacity: 0;
      transition: opacity 0.15s ease;
      font-weight: normal;
    }

    .doc-heading:hover .heading-anchor {
      opacity: 1;
      color: var(--color-primary);
    }

    /* Paragraphs & Text */
    p {
      margin-bottom: 1.2em;
      color: var(--text-main);
    }

    strong {
      color: var(--text-heading);
      font-weight: 600;
    }

    hr {
      border: none;
      height: 1px;
      background: var(--border-subtle);
      margin: 3em 0;
    }

    /* Lists */
    ul, ol {
      margin-bottom: 1.2em;
      padding-left: 1.8em;
    }

    li {
      margin-bottom: 0.5em;
    }

    li::marker {
      color: var(--color-primary);
    }

    /* Blockquotes */
    blockquote {
      border-left: 4px solid var(--color-primary);
      background: rgba(56, 189, 248, 0.05);
      padding: 14px 20px;
      border-radius: 0 var(--radius-md) var(--radius-md) 0;
      margin: 1.5em 0;
      color: var(--text-main);
    }

    [data-theme="light"] blockquote {
      background: rgba(2, 132, 199, 0.06);
    }

    blockquote p:last-child {
      margin-bottom: 0;
    }

    /* Enhanced Inline Code Chips */
    :not(pre) > code {
      font-family: var(--font-mono);
      font-size: 0.88em;
      padding: 2px 7px;
      border-radius: 5px;
      background: rgba(56, 189, 248, 0.09);
      color: #38bdf8;
      border: 1px solid rgba(56, 189, 248, 0.22);
      font-weight: 500;
      white-space: nowrap;
    }

    [data-theme="light"] :not(pre) > code {
      background: #f1f5f9;
      color: #0284c7;
      border-color: #cbd5e1;
    }

    /* Code Block Cards with Professional Highlighting */
    .code-card {
      background: var(--bg-code);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      margin: 1.8em 0;
      overflow: hidden;
      box-shadow: 0 4px 18px rgba(0, 0, 0, 0.3);
      transition: border-color 0.2s ease;
    }

    .code-card:hover {
      border-color: rgba(56, 189, 248, 0.35);
    }

    [data-theme="light"] .code-card {
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
    }

    .code-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 16px;
      background: var(--bg-code-header);
      border-bottom: 1px solid var(--border-subtle);
    }

    .code-dots {
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
    }

    .dot-red { background: #ff5f56; }
    .dot-yellow { background: #ffbd2e; }
    .dot-green { background: #27c93f; }

    .code-meta-wrap {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .code-lang-badge {
      font-size: 0.68rem;
      font-family: var(--font-mono);
      font-weight: 700;
      letter-spacing: 0.06em;
      padding: 2px 7px;
      border-radius: 4px;
      text-transform: uppercase;
    }

    .code-lang-badge.lang-python {
      background: rgba(56, 189, 248, 0.15);
      color: #38bdf8;
      border: 1px solid rgba(56, 189, 248, 0.3);
    }

    .code-lang-badge.lang-bash, .code-lang-badge.lang-sh {
      background: rgba(16, 185, 129, 0.15);
      color: #10b981;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .code-lang-badge.lang-json {
      background: rgba(245, 158, 11, 0.15);
      color: #f59e0b;
      border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .code-lang-badge.lang-text {
      background: rgba(148, 163, 184, 0.15);
      color: #94a3b8;
      border: 1px solid rgba(148, 163, 184, 0.3);
    }

    .code-line-count {
      font-size: 0.72rem;
      color: var(--text-dim);
      font-family: var(--font-mono);
    }

    .copy-code-btn {
      display: flex;
      align-items: center;
      gap: 5px;
      padding: 4px 10px;
      border-radius: var(--radius-sm);
      background: transparent;
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      font-size: 0.75rem;
      cursor: pointer;
      transition: var(--transition-base);
    }

    .copy-code-btn:hover {
      background: rgba(255, 255, 255, 0.08);
      color: var(--text-heading);
      border-color: var(--color-primary);
    }

    .copy-code-btn.copied {
      color: var(--color-success);
      border-color: var(--color-success);
      background: rgba(16, 185, 129, 0.1);
    }

    /* Code Viewport with Gutter Line Numbers */
    .code-viewport {
      padding: 12px 0;
      overflow-x: auto;
    }

    .code-lines {
      display: flex;
      flex-direction: column;
      font-family: var(--font-mono);
      font-size: 0.88rem;
      line-height: 1.68;
    }

    .code-line {
      display: flex;
      min-height: 1.68em;
      transition: background-color 0.12s ease;
    }

    .code-line:hover {
      background-color: rgba(255, 255, 255, 0.045);
    }

    [data-theme="light"] .code-line:hover {
      background-color: rgba(0, 0, 0, 0.035);
    }

    .line-num {
      width: 46px;
      flex-shrink: 0;
      padding-right: 18px;
      text-align: right;
      color: var(--text-dim);
      user-select: none;
      font-size: 0.78rem;
      opacity: 0.55;
    }

    .line-text {
      flex: 1;
      padding-right: 24px;
      white-space: pre;
      font-family: inherit;
    }

    /* Syntax Highlighting Tokens (Dark Mode) */
    [data-theme="dark"] .hljs-keyword,
    [data-theme="dark"] .hl-kw {
      color: #c678dd;
      font-weight: 600;
    }

    [data-theme="dark"] .hljs-function,
    [data-theme="dark"] .hljs-title.function_,
    [data-theme="dark"] .calc-func {
      color: #61afef;
      font-weight: 600;
    }

    [data-theme="dark"] .hljs-title.class_,
    [data-theme="dark"] .hljs-type {
      color: #e5c07b;
      font-weight: 600;
    }

    [data-theme="dark"] .hljs-string,
    [data-theme="dark"] .hl-str {
      color: #98c379;
    }

    [data-theme="dark"] .hljs-number,
    [data-theme="dark"] .calc-num {
      color: #d19a66;
    }

    [data-theme="dark"] .hljs-comment,
    [data-theme="dark"] .hl-cmt {
      color: #7f848e;
      font-style: italic;
    }

    [data-theme="dark"] .hljs-built_in {
      color: #56b6c2;
    }

    [data-theme="dark"] .hljs-attr {
      color: #e06c75;
      font-weight: 600;
    }

    [data-theme="dark"] .hljs-literal {
      color: #d19a66;
      font-weight: 600;
    }

    [data-theme="dark"] .calc-var {
      color: #e5c07b;
      font-weight: 600;
    }

    [data-theme="dark"] .calc-note {
      color: #94a3b8;
      font-style: italic;
    }

    /* CLI & Terminal Tokens (Dark Mode) */
    [data-theme="dark"] .cmd-name {
      color: #e5c07b;
      font-weight: 700;
    }

    [data-theme="dark"] .cmd-path {
      color: #61afef;
    }

    [data-theme="dark"] .cmd-flag {
      color: #c678dd;
      font-weight: 600;
    }

    [data-theme="dark"] .cmd-arg {
      color: #98c379;
      font-weight: 600;
    }

    [data-theme="dark"] .term-stars {
      color: #fbbf24;
      text-shadow: 0 0 8px rgba(251, 191, 36, 0.5);
      font-weight: bold;
    }

    [data-theme="dark"] .term-tier-badge {
      color: #34d399;
      font-weight: bold;
      text-shadow: 0 0 8px rgba(52, 211, 153, 0.4);
    }

    [data-theme="dark"] .term-tier-sub {
      color: #38bdf8;
    }

    [data-theme="dark"] .term-main-title {
      color: #38bdf8;
      font-weight: bold;
      letter-spacing: 0.05em;
    }

    [data-theme="dark"] .term-dim-title {
      color: #a78bfa;
      font-weight: 600;
    }

    /* Syntax Highlighting Tokens (Light Mode) */
    [data-theme="light"] .hljs-keyword,
    [data-theme="light"] .hl-kw {
      color: #a626a4;
      font-weight: 600;
    }

    [data-theme="light"] .hljs-function,
    [data-theme="light"] .hljs-title.function_,
    [data-theme="light"] .calc-func {
      color: #4078f2;
      font-weight: 600;
    }

    [data-theme="light"] .hljs-title.class_,
    [data-theme="light"] .hljs-type {
      color: #c18401;
      font-weight: 600;
    }

    [data-theme="light"] .hljs-string,
    [data-theme="light"] .hl-str {
      color: #50a14f;
    }

    [data-theme="light"] .hljs-number,
    [data-theme="light"] .calc-num {
      color: #986801;
    }

    [data-theme="light"] .hljs-comment,
    [data-theme="light"] .hl-cmt {
      color: #a0a1a7;
      font-style: italic;
    }

    [data-theme="light"] .hljs-built_in {
      color: #0184bc;
    }

    [data-theme="light"] .hljs-attr {
      color: #e45649;
      font-weight: 600;
    }

    [data-theme="light"] .hljs-literal {
      color: #0184bc;
      font-weight: 600;
    }

    [data-theme="light"] .calc-var {
      color: #986801;
      font-weight: 600;
    }

    [data-theme="light"] .calc-note {
      color: #64748b;
      font-style: italic;
    }

    [data-theme="light"] .cmd-name {
      color: #b45309;
      font-weight: 700;
    }

    [data-theme="light"] .cmd-path {
      color: #2563eb;
    }

    [data-theme="light"] .cmd-flag {
      color: #7c3aed;
      font-weight: 600;
    }

    [data-theme="light"] .cmd-arg {
      color: #15803d;
      font-weight: 600;
    }

    [data-theme="light"] .term-stars {
      color: #d97706;
      font-weight: bold;
    }

    [data-theme="light"] .term-tier-badge {
      color: #059669;
      font-weight: bold;
    }

    [data-theme="light"] .term-tier-sub {
      color: #0284c7;
    }

    [data-theme="light"] .term-main-title {
      color: #0284c7;
      font-weight: bold;
    }

    [data-theme="light"] .term-dim-title {
      color: #6d28d9;
      font-weight: 600;
    }

    /* Mermaid Cards */
    .mermaid-block-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-xl);
      margin: 2em 0;
      overflow: hidden;
      box-shadow: var(--shadow-md);
    }

    .mermaid-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 20px;
      background: var(--bg-surface-elevated);
      border-bottom: 1px solid var(--border-subtle);
    }

    .mermaid-title-wrap {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .mermaid-icon {
      font-size: 1.1rem;
    }

    .mermaid-title {
      font-size: 0.88rem;
      font-weight: 600;
      color: var(--text-heading);
    }

    .mermaid-actions {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .action-btn {
      padding: 4px 10px;
      border-radius: var(--radius-sm);
      background: transparent;
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      font-size: 0.72rem;
      cursor: pointer;
      transition: var(--transition-base);
    }

    .action-btn:hover {
      background: var(--bg-card-hover);
      color: var(--text-heading);
      border-color: var(--color-primary);
    }

    .mermaid-viewport {
      padding: 24px;
      display: flex;
      justify-content: center;
      background: var(--bg-body);
      overflow-x: auto;
    }

    .mermaid-viewport svg {
      max-width: 100%;
      height: auto;
    }

    /* Table Cards */
    .table-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-xl);
      margin: 2em 0;
      overflow: hidden;
      box-shadow: var(--shadow-sm);
    }

    .table-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 18px;
      background: var(--bg-surface-elevated);
      border-bottom: 1px solid var(--border-subtle);
      gap: 12px;
    }

    .table-tag {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-muted);
    }

    .table-search-box {
      width: 240px;
    }

    .table-filter-input {
      width: 100%;
      padding: 6px 12px;
      border-radius: var(--radius-md);
      background: var(--bg-body);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      font-size: 0.78rem;
      outline: none;
      transition: var(--transition-base);
    }

    .table-filter-input:focus {
      border-color: var(--color-primary);
      box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.15);
    }

    .table-responsive {
      width: 100%;
      overflow-x: auto;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.86rem;
      text-align: left;
    }

    thead {
      background: var(--bg-table-header);
      position: sticky;
      top: 0;
      z-index: 5;
    }

    th {
      padding: 12px 16px;
      font-weight: 600;
      color: var(--text-heading);
      border-bottom: 2px solid var(--border-table);
      white-space: nowrap;
    }

    td {
      padding: 12px 16px;
      border-bottom: 1px solid var(--border-table);
      color: var(--text-main);
      vertical-align: top;
    }

    tbody tr:nth-child(even) {
      background: var(--bg-table-row-alt);
    }

    tbody tr:hover {
      background: var(--bg-table-row-hover);
    }

    /* Badges */
    .badge {
      display: inline-block;
      font-size: 0.72rem;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 999px;
      line-height: 1.4;
      white-space: nowrap;
    }

    .badge-success {
      background: rgba(16, 185, 129, 0.15);
      color: var(--color-success);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .badge-danger {
      background: rgba(244, 63, 94, 0.15);
      color: var(--color-danger);
      border: 1px solid rgba(244, 63, 94, 0.3);
    }

    .badge-warning {
      background: rgba(245, 158, 11, 0.15);
      color: var(--color-warning);
      border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .badge-primary {
      background: rgba(56, 189, 248, 0.15);
      color: var(--color-primary);
      border: 1px solid rgba(56, 189, 248, 0.3);
    }

    .num-positive {
      color: #34d399;
      font-weight: 600;
      font-family: var(--font-mono);
    }

    .num-negative {
      color: #f87171;
      font-weight: 600;
      font-family: var(--font-mono);
    }

    /* Floating Utilities */
    .floating-tools {
      position: fixed;
      right: 24px;
      bottom: 24px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      z-index: 800;
    }

    .float-btn {
      width: 44px;
      height: 44px;
      border-radius: 50%;
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      box-shadow: var(--shadow-md);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.1rem;
      transition: var(--transition-base);
    }

    .float-btn:hover {
      background: var(--color-primary);
      color: #000;
      transform: translateY(-3px);
      box-shadow: var(--shadow-glow);
    }

    /* Search Modal Overlay */
    .search-modal-overlay {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0, 0, 0, 0.7);
      backdrop-filter: blur(8px);
      z-index: 2000;
      display: none;
      align-items: flex-start;
      justify-content: center;
      padding-top: 12vh;
    }

    .search-modal {
      width: 100%;
      max-width: 640px;
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-xl);
      box-shadow: var(--shadow-lg);
      overflow: hidden;
      display: flex;
      flex-direction: column;
      animation: modalFadeIn 0.2s ease;
    }

    @keyframes modalFadeIn {
      from { opacity: 0; transform: translateY(-12px) scale(0.98); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }

    .search-input-wrap {
      display: flex;
      align-items: center;
      padding: 14px 18px;
      border-bottom: 1px solid var(--border-subtle);
      gap: 10px;
    }

    .search-input-wrap svg {
      color: var(--text-dim);
    }

    .modal-search-input {
      flex: 1;
      background: transparent;
      border: none;
      outline: none;
      font-size: 1.1rem;
      color: var(--text-heading);
    }

    .modal-close-kbd {
      padding: 3px 6px;
      border-radius: var(--radius-sm);
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      font-size: 0.75rem;
      color: var(--text-dim);
      cursor: pointer;
    }

    .search-results-list {
      max-height: 400px;
      overflow-y: auto;
      padding: 10px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .search-result-item {
      padding: 10px 14px;
      border-radius: var(--radius-md);
      cursor: pointer;
      text-decoration: none;
      display: flex;
      flex-direction: column;
      gap: 3px;
      transition: var(--transition-base);
    }

    .search-result-item:hover, .search-result-item.selected {
      background: rgba(56, 189, 248, 0.12);
    }

    .result-title {
      font-size: 0.92rem;
      font-weight: 600;
      color: var(--text-heading);
    }

    .result-snippet {
      font-size: 0.78rem;
      color: var(--text-muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    /* Print Style Optimization */
    @media print {
      .top-navbar, .doc-sidebar, .floating-tools, .copy-code-btn, .table-toolbar {
        display: none !important;
      }
      .doc-main {
        margin-left: 0 !important;
        padding: 0 !important;
        max-width: 100% !important;
      }
      body {
        background: #fff !important;
        color: #000 !important;
      }
      .code-card {
        border: 1px solid #ccc !important;
        background: #f8f9fa !important;
        color: #000 !important;
      }
    }

    /* Responsive */
    @media (max-width: 1024px) {
      .doc-sidebar {
        transform: translateX(-100%);
      }
      .doc-sidebar.open {
        transform: translateX(0);
        box-shadow: 0 0 30px rgba(0, 0, 0, 0.7);
      }
      .doc-main {
        margin-left: 0;
        max-width: 100%;
        padding: 30px 20px;
      }
      .mobile-menu-btn {
        display: flex !important;
      }
    }

    .mobile-menu-btn {
      display: none;
    }

    /* Toast Notification */
    .toast-msg {
      position: fixed;
      bottom: 30px;
      left: 50%;
      transform: translateX(-50%) translateY(50px);
      padding: 10px 20px;
      background: var(--bg-surface-elevated);
      color: var(--text-heading);
      border: 1px solid var(--border-subtle);
      border-radius: 999px;
      font-size: 0.85rem;
      box-shadow: var(--shadow-lg);
      z-index: 3000;
      opacity: 0;
      pointer-events: none;
      transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .toast-msg.show {
      transform: translateX(-50%) translateY(0);
      opacity: 1;
    }
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
            <span class="meta-tag purple">11 大认知阶梯</span>
            <span class="meta-tag green">31 项核心超参</span>
            <span class="meta-tag">29 项严密指标</span>
            <span class="meta-tag purple">4000+ 行实战引擎</span>
          </div>
          <h1>Kronos-Pro 量化回测系统完全解析与认知实战手册</h1>
          <p class="doc-hero-desc">
            从第一性原理、微观物理摩擦、因果时序律，到 AI 预测特征解耦、九重风险否决、离散状态机出场与 300418 端到端实战数值穿透。专为量化研发、高频回测与工业化实盘部署设计。
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

    // Reading Progress & Scroll Spy
    window.addEventListener('scroll', () => {
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      const scrolled = window.scrollY;
      const pct = docHeight > 0 ? (scrolled / docHeight) * 100 : 0;
      document.getElementById('reading-progress-bar').style.width = pct + '%';

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
