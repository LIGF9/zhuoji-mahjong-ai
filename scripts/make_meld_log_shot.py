"""生成「副露/手牌几何 + 流水面板」截图夹具，用于像素级核对 UI 口径。

核对目标（2026-10-11 用户提出）：
  1. 碰鸡横置牌：右侧相邻牌对横牌的遮盖量是否过大；
  2. 横置牌的侧边与竖牌的底边是否对齐；
  3. 手牌（68×104）与副露牌（53×82）尺寸不同，但底边是否对齐；
  4. 流水面板：碰/杠/报叫/听牌/胡 的高亮样式与可缩放把手。

构造场景（我 = seat 0）：
  * 手牌 11 张 + 右侧刚摸进 1 张（与手牌之间有 drawsep）；
  * 副露 4 组，覆盖三种待核对情形：
      - 普通碰（三张竖排，无横置）—— 底边对齐基准；
      - 碰横鸡（横置牌带「横」蓝角标），横置牌分别位于组首 / 组尾；
      - 明杠（横置牌带角标）。
  * 流水面板展开，塞入含 碰/杠/报叫/听牌/胡/摸牌/打牌/结算 的样例事件。

产物：``web/static/assets/_shot_meld_log.html``，用真实服务器访问
``http://127.0.0.1:8770/assets/_shot_meld_log.html`` 即可（/api/meta 放行走真实接口）。

无头核对方式：
  * 截图 —— 直接看视觉效果；
  * ``--dump-dom`` —— 几何数据写在 ``<title>`` 里（``__GEOM__{...}``），
    比页面上叠一层 ``<pre>`` 更容易被可靠抓取（旧的覆盖层方案在无头下从未生效）。
  * URL 带 ``#dbg`` —— 额外加盒模型描边（黄=.meld 青=.mc 红=.mc.heng 品红=.tile-mine）。

用法::

    python scripts/make_meld_log_shot.py [out.html]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web/static/assets" / (
    sys.argv[1] if len(sys.argv) > 1 else "_shot_meld_log.html")

src = (ROOT / "web/static/dushan.html").read_text(encoding="utf-8")
fixtures = json.loads((ROOT / "reports/ui_fixtures_over.json").read_text(encoding="utf-8"))
state = fixtures[0]["state"]
state["over"] = False

inject = """<script>
/* 顶层错误陷阱：把注入脚本的异常直接打到页面上，便于无头环境排错 */
window.__errs = [];
window.onerror = function(msg, src, line, col, err){
  window.__errs.push(msg + ' @' + line + ':' + col + (err && err.stack ? '\\n' + err.stack : ''));
  var d = document.getElementById('injerr');
  if (!d){ d = document.createElement('pre'); d.id = 'injerr';
    d.style.cssText = 'position:fixed;left:8px;top:8px;z-index:10000;background:#7a1010;color:#fff;'
      + 'font:11px/1.4 monospace;padding:8px 10px;border-radius:8px;white-space:pre-wrap;max-width:60vw;margin:0';
    document.body.appendChild(d); }
  d.textContent = window.__errs.join('\\n\\n');
};
/* 只挂起 /api/state（防止真实轮询覆盖注入态）；/api/meta 放行，TILES 才能填上 */
(function(){
  var orig = window.fetch;
  window.fetch = function(url, opt){
    if (String(url).indexOf('/api/state') >= 0) return new Promise(function(){});
    return orig.apply(this, arguments);
  };
})();
var __ST = %s;
setTimeout(function(){
  var b = document.getElementById('boot');
  if (b) b.style.display = 'none';
  var st = JSON.parse(JSON.stringify(__ST));
  st.my_seat = 0;
  st.over = false;
  st.turn = 0;
  st.phase = 'discard';
  st.players = [0,1,2,3].map(function(i){ var p = st.players[i] || {}; p.seat = i; p.rel = i; return p; });
  st.ji_tiles = [9, 20];

  // ---- 我的手牌：11 张 + 右侧摸进 1 张 ----
  st.hand = [0,1,2,3,4,5,6,7,10,11,20];
  st.last_draw = 20;
  // 注意：pending 必须是 null，或带 buttons 数组的真实结构。
  // 曾写成 { mode, discardables }（过时口径）→ renderPending 里 pend.buttons.map 抛
  // TypeError，render() 中途中断，其后的 renderLog 根本没跑（夹具流水一直是空的）。
  st.pending = null;

  // ---- 我的副露：覆盖各种待核对情形 ----
  // 1) 碰横鸡，横置牌在组首（src = 上家）
  // 2) 碰横鸡，横置牌在组尾（src = 对家）—— 右侧相邻牌遮盖的极端情形
  // 3) 明杠幺鸡，横置牌在组内（src = 下家）
  // 4) 普通碰，无横置牌（底边对齐基准）
  st.players[0].melds = [
    { type: 1, label: '碰', tile: 11, src: 3, ji: '横鸡' },
    { type: 1, label: '碰', tile: 5,  src: 2, ji: '横鸡' },
    { type: 2, label: '明杠', tile: 9, src: 1, ji: '幺鸡' },
    { type: 1, label: '碰', tile: 2,  src: 2, ji: null }
  ];
  st.players[0].hand_count = 11;

  // ---- 对家/左右家简单摆一点，便于整体观察 ----
  st.players[1].melds = [{ type: 1, label: '碰', tile: 3, src: 2, ji: null }];
  st.players[2].melds = [{ type: 1, label: '碰', tile: 7, src: 3, ji: '横鸡' }];
  st.players[3].melds = [];

  // ---- 流水样例事件：kind 与文案严格对齐服务端 _emit_action / _emit_progress 的实际产出 ----
  st.events = [
    { t: 1.0, kind: 'result',  rel: 0, text: '上一局：大师 胡牌 · 平胡 · +8 分', tile: null, ctag: '' },
    { t: 2.0, kind: 'draw',    rel: 0, text: '你 摸进 三万', tile: 2, ctag: '' },
    { t: 2.5, kind: 'discard', rel: 0, text: '你 打出 九万', tile: 8, ctag: '' },
    { t: 3.0, kind: 'draw',    rel: 1, text: '大师 摸进 五条', tile: 13, ctag: '' },
    { t: 3.4, kind: 'peng',    rel: 2, text: '教师·均衡 碰 六万', tile: 5, ctag: '' },
    { t: 4.0, kind: 'baojiao', rel: 2, text: '教师·均衡 报叫！', tile: null, ctag: '报叫' },
    { t: 4.6, kind: 'draw',    rel: 3, text: '随机 摸进 一条', tile: 9, ctag: '' },
    { t: 5.0, kind: 'kong',    rel: 1, text: '大师 补杠七条（得通行证）', tile: 20, ctag: '' },
    { t: 5.4, kind: 'discard', rel: 0, text: '你 打出 三条（鸡）', tile: 11, ctag: '横鸡' },
    { t: 5.8, kind: 'ting',    rel: 0, text: '你 听牌', tile: null, ctag: '' },
    { t: 6.2, kind: 'draw',    rel: 2, text: '教师·均衡 摸牌', tile: null, ctag: '' },
    { t: 6.6, kind: 'discard', rel: 2, text: '教师·均衡 打出 七万（热炮）', tile: 6, ctag: '' },
    { t: 7.0, kind: 'hu',      rel: 1, text: '大师 胡牌！', tile: null, ctag: '' },
    { t: 7.4, kind: 'result',  rel: 1, text: '大师 胡牌 · 大对子 · +16 分', tile: null, ctag: '' }
  ];
  LAST_STATE = st;
  render(st);

  // 流水面板默认收起：这里强制展开，并拉高让全部样例行都进入可视区
  var lg = document.getElementById('log');
  if (lg){ lg.classList.remove('hidden'); lg.style.height = '340px'; lg.style.maxHeight = 'none'; }
  if (typeof applyLogBtn === 'function') applyLogBtn();

  // ---- 几何测量：结果写入 <title>（无头 --dump-dom 可稳定抓取）----
  (function measure(){
    function box(e){ var r = e.getBoundingClientRect();
      return { x: r.left, w: r.width, h: r.height, b: r.bottom }; }
    function fmt(v){ return Math.round(v * 10) / 10; }
    var out = { melds: [], hand: null, log: null };

    // 我的副露：按 .meld 分组，组内每格与「该组竖牌基准」比底边
    document.querySelectorAll('#me-melds .meld').forEach(function(m, gi){
      var cells = m.querySelectorAll('.mc');
      var base = null;
      cells.forEach(function(c){
        if (!c.classList.contains('heng') && base === null) base = box(c).b;
      });
      if (base === null && cells.length) base = box(cells[0]).b;
      cells.forEach(function(c, ci){
        var r = box(c);
        out.melds.push({ g: gi, i: ci, heng: c.classList.contains('heng'),
          x: fmt(r.x), w: fmt(r.w), h: fmt(r.h), bottom: fmt(r.b), dGroup: fmt(r.b - base) });
      });
    });

    // 我的手牌
    var hand = document.querySelectorAll('#hand .tile-mine:not(.ghost)');
    if (hand.length){
      var hb = box(hand[0]).b;
      out.hand = { base: fmt(hb), n: hand.length, first: fmt(box(hand[0]).w),
        h: fmt(box(hand[0]).h), last: fmt(box(hand[hand.length-1]).b) };
    }

    // 副露整体 vs 手牌整体的底边差（第 4 项核对的核心数字）
    var mm = document.getElementById('me-melds');
    var hd = document.getElementById('hand');
    if (mm && hd){
      out.rowDelta = fmt(box(mm).b - box(hd).b);
      out.meldsBox = { w: fmt(box(mm).w), h: fmt(box(mm).h), bottom: fmt(box(mm).b) };
      out.handBox = { w: fmt(box(hd).w), h: fmt(box(hd).h), bottom: fmt(box(hd).b) };
    }

    // 流水面板
    var lgd = document.getElementById('log');
    var lgb = document.getElementById('log-body');
    var grip = document.getElementById('log-rsz');
    if (lgd){
      out.log = { w: fmt(box(lgd).w), h: fmt(box(lgd).h),
        visible: !lgd.classList.contains('hidden'),
        bodyScrollH: lgb ? lgb.scrollHeight : null,
        bodyClientH: lgb ? lgb.clientHeight : null,
        grip: !!grip, gripCursor: grip ? getComputedStyle(grip).cursor : null,
        lines: document.querySelectorAll('#log-body .le').length,
        kinds: [].slice.call(document.querySelectorAll('#log-body .le'))
                   .map(function(d){ return d.className.replace('le ', '').trim() || '-'; }) };
    }
    out.errs = window.__errs;
    out.vw = innerWidth; out.vh = innerHeight;
    var cs = getComputedStyle(document.documentElement);
    out.tileW = cs.getPropertyValue('--tile-w').trim();
    out.mob = document.documentElement.className;
    var gg = document.getElementById('gate');
    out.gate = gg ? { display: getComputedStyle(gg).display,
                      z: getComputedStyle(gg).zIndex,
                      rect: (function(){ var r = gg.getBoundingClientRect();
                        return [r.x|0, r.y|0, r.width|0, r.height|0]; })() }
                   : 'MISSING';
    try {
      out.dbg = { melds: (window.LAST_STATE && LAST_STATE.players[0].melds || []).length,
        handN: window.LAST_STATE ? LAST_STATE.hand.length : null,
        fn: typeof window.fitMyRow };
    } catch (e) { out.dbg = 'dbgERR ' + e; }
    var mr = document.getElementById('me-row');
    if (mr) out.meRow = { x: fmt(box(mr).x), w: fmt(box(mr).w), b: fmt(box(mr).b) };
    var meb = document.getElementById('me');
    if (meb) out.meBox = { w: fmt(box(meb).w) };
    document.title = '__GEOM__' + JSON.stringify(out);
  })();
}, 700);
</script>
""" % json.dumps(state, ensure_ascii=False)

assert "</body>" in src, "未找到 </body>"
OUT.write_text(src.replace("</body>", inject + "</body>", 1), encoding="utf-8")
print("wrote", OUT)
