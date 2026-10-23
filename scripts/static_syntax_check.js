/* static_syntax_check.js —— 页面静态资源完整性闸门（CSS / JS 注释与括号）
 *
 * 背景（2026-10-12）
 * -----------------
 * `dushan.html` 是单文件 SPA：内联了一个 `<style>` 和一段 `<script>`。用字符串替换
 * 改这份文件时很容易**吃掉一个注释开头**（`/* ===== xxx =====`）—— 断言里的 CSS/JS
 * 就从注释变成裸文本，浏览器会一路吞掉后续规则/代码直到下一个 `*​/`，
 * 表现是「只改了一处，别的样式全错位」。而 jsdom 的结构断言**查不出来**（它只看 DOM），
 * `new vm.Script()` 也查不出来（多出来的东西正好是合法语句或全被注释掉）。
 * 所以这里做的是**词法级**配平检查。
 *
 * 检查项：
 *   1) CSS：逐字符扫描（注释 / 字符串 / 花括号）→ 断言全部配平
 *   2) JS ：同上，另外识别模板串（含 `${}` 插值嵌套）与正则字面量 → 断言块注释配平
 *   3) JS ：用 vm 编译一遍，断言无语法错误
 *
 * 修订（2026-10-12 晚）
 * -------------------
 * 旧版扫描器有两个已知误报，会在页尾报「多余的 } 」和「<style> 不在最前」：
 *   a. 模板串：旧版遇到 `` ` `` 就一路吃到下一个反引号，**不会**处理 `${ ... }`
 *      插值——插值里的 `}` 被当成「多余的花括号」。页里嵌套模板（`` `${c ? `<span>` : ""}` ``）
 *      一多，漂移就累积到文件末尾才爆。
 *   b. `/` 误判为正则：漂移落进模板/字符串后，`/` 被 `isReStart` 判成正则开头，
 *      吞掉一段代码里的 `{` `}`，进一步加剧漂移。
 *   c. `<style>` 位置：`document.write('<style>' + ff + '</style>')` 这句**字符串**里的
 *      `<style>` 被当成真样式块，且 `indexOf('<style')` 落在页首 `<script>` 之后 → 误 FAIL。
 * 现在改为：扫描器按**状态栈**处理模板插值（进入 `${` 压栈、`}` 回到模板态），
 * 正则扫描带字符组 `[...]` 支持且不跨行；样式块提取会**排除落在 `<script>` 区间内**的匹配；
 * 入口结构改为比较「主」`<style>`（体积最大者）与「主」`<script>` 的先后。
 *
 * 运行：
 *   NODE_PATH=C:/Users/LIGF/.workbuddy/binaries/node/workspace/node_modules \
 *     node scripts/static_syntax_check.js
 */
const fs = require('fs');
const vm = require('vm');

/* ROOT 可用 DUSHAN_HTML 覆盖（仅文件名），便于对「故意注入缺陷的副本」做反例自检。 */
const ROOT = 'C:/Users/LIGF/WorkBuddy/2026-09-30-12-49-43/zhuoji-mahjong';
const HTML_PATH = process.env.DUSHAN_HTML || (ROOT + '/web/static/dushan.html');
const html = fs.readFileSync(HTML_PATH, 'utf8');
const lineOf = (idx) => html.slice(0, idx).split('\n').length;

let pass = 0, fail = 0;
function check(name, ok, extra = '') {
  if (ok) { pass++; console.log(`  PASS  ${name}${extra ? '  ' + extra : ''}`); }
  else { fail++; console.log(`  FAIL  ${name}${extra ? '  ' + extra : ''}`); }
}

/* 先取 <script>，再取**不在 script 区间内**的 <style>：
   否则 `document.write('<style>' + ff + '</style>')` 这类字符串会被误当成样式块。 */
const scriptBlocks = [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)];
const scriptRanges = scriptBlocks.map((m) => [m.index, m.index + m[0].length]);
const inScript = (idx) => scriptRanges.some(([a, b]) => idx >= a && idx < b);
const styleBlocks = [...html.matchAll(/<style[^>]*>([\s\S]*?)<\/style>/g)]
  .filter((m) => !inScript(m.index));

/* ---- 通用词法扫描 ----------------------------------------------------
   mode: 'css' 只认注释 / 字符串 / 花括号；'js' 额外认模板串与正则字面量。
   返回 bad[] —— 人话错误列表。

   状态：'code' | 'line'（行注释）| 'str' | 'tpl'
   - 模板串插值：在 'tpl' 里遇到 `${` → 压入 tplExpr（记下当时的花括号栈深），
     回到 'code'；在 'code' 里遇到 `}` 且花括号栈深 == tplExpr 顶部 → 弹栈，回到 'tpl'。
   - 正则字面量：仅在 isReStart(prevSig) 成立时按正则吃；带字符组 `[...]` 支持。 */
function scan(src, mode, baseLine) {
  const bad = [];
  let prevSig = '';        // 上一个有意义字符（判定 `/` 是否开正则）

  let i = 0, line = 1;
  let state = 'code';
  let strQ = '';
  let inCmt = false;
  const braceStack = [];   // 每个 '{' 的行号
  const tplExpr = [];      // 每个模板插值 `${` 进入时的 braceStack.length

  const reStart = (c) => '(,=:[!&|?{};+-*%~^<>\n'.indexOf(c) >= 0 || prevSig === '';

  while (i < src.length) {
    const c = src[i], nxt = src[i + 1] || '';
    if (c === '\n') line++;

    if (inCmt) {
      if (c === '*' && nxt === '/') { inCmt = false; i += 2; prevSig = '/'; continue; }
      i++; continue;
    }
    if (state === 'line') {
      if (c === '\n') state = 'code';
      i++; continue;
    }
    if (state === 'str') {
      if (c === '\\') { i += 2; continue; }
      if (c === strQ) { state = 'code'; prevSig = c; }
      i++; continue;
    }
    if (state === 'tpl') {
      if (c === '\\') { i += 2; continue; }
      if (c === '`') { state = 'code'; prevSig = '`'; i++; continue; }
      if (mode === 'js' && c === '$' && nxt === '{') {
        tplExpr.push(braceStack.length);
        state = 'code'; prevSig = '{'; i += 2; continue;
      }
      i++; continue;
    }

    /* ---- code 态 ---- */
    if (c === '/' && nxt === '*') { inCmt = true; i += 2; continue; }
    if (c === '/' && nxt === '/') { state = 'line'; i += 2; continue; }
    if (c === '"' || c === "'") { strQ = c; state = 'str'; prevSig = c; i++; continue; }
    if (mode === 'js' && c === '`') { state = 'tpl'; prevSig = '`'; i++; continue; }
    if (mode === 'js' && c === '/' && reStart(prevSig)) {
      i++;                                  // 跳过开头的 /
      while (i < src.length) {
        const d = src[i];
        if (d === '\\') { i += 2; continue; }
        if (d === '\n') break;              // 正则不跨行：遇换行交回主循环
        if (d === '/') { i++; break; }
        if (d === '[') {                    // 字符组内的 / 不结束正则
          i++;
          while (i < src.length && src[i] !== ']') {
            if (src[i] === '\\') i++;
            if (src[i] === '\n') line++;
            i++;
          }
          i++; continue;
        }
        i++;
      }
      prevSig = '/'; continue;
    }
    if (c === '{') { braceStack.push(line); prevSig = '{'; i++; continue; }
    if (c === '}') {
      if (tplExpr.length && braceStack.length === tplExpr[tplExpr.length - 1]) {
        tplExpr.pop(); state = 'tpl'; prevSig = '}'; i++; continue;   // 关闭插值，回到模板
      }
      if (braceStack.length === 0) {
        bad.push(`多余的 } （第 ${baseLine + line - 1} 行）`);
      } else {
        braceStack.pop();
      }
      prevSig = '}'; i++; continue;
    }
    if (!/\s/.test(c)) prevSig = c;
    i++;
  }

  if (inCmt) bad.push(`未闭合块注释（该段起自第 ${baseLine} 行）`);
  if (state === 'str') bad.push(`未闭合字符串 ${strQ}（该段起自第 ${baseLine} 行）`);
  if (state === 'tpl') bad.push(`未闭合模板串（该段起自第 ${baseLine} 行）`);
  if (braceStack.length !== 0) bad.push(`花括号未配平，剩余 ${braceStack.length} 个 {`);
  return bad;
}

console.log('=== 1. CSS 词法配平 ===');
check('页面含内联 <style>', styleBlocks.length > 0, `${styleBlocks.length} 个`);
let cssBad = [];
for (const m of styleBlocks) cssBad = cssBad.concat(scan(m[1], 'css', lineOf(m.index)));
check('CSS 注释 / 字符串 / 花括号全部配平', cssBad.length === 0, cssBad.slice(0, 3).join(' | '));

console.log('\n=== 2. JS 词法配平 ===');
check('页面含内联 <script>', scriptBlocks.length > 0, `${scriptBlocks.length} 段`);
let jsBad = [];
for (const m of scriptBlocks) jsBad = jsBad.concat(scan(m[1], 'js', lineOf(m.index)));
check('JS 块注释 / 字符串 / 模板串全部配平', jsBad.length === 0, jsBad.slice(0, 3).join(' | '));

console.log('\n=== 3. <script> 可编译 ===');
const jsCompileBad = [];
scriptBlocks.forEach((m, n) => {
  try {
    new vm.Script(m[1], { filename: `dushan.html:script#${n + 1}` });
  } catch (e) {
    jsCompileBad.push(`第 ${n + 1} 段（起自第 ${lineOf(m.index)} 行）: ` +
      String(e.message).split('\n')[0]);
  }
});
check('全部 <script> 无语法错误', jsCompileBad.length === 0, jsCompileBad.slice(0, 3).join(' | '));

console.log('\n=== 4. 入口结构 ===');
const bigStyle = [...styleBlocks].sort((a, b) => b[1].length - a[1].length)[0];
const bigScript = [...scriptBlocks].sort((a, b) => b[1].length - a[1].length)[0];
for (const [what, cond] of [
  ['主 <style> 早于主 <script>（避免 FOUC）', !!bigStyle && !!bigScript && bigStyle.index < bigScript.index],
  ['<body> 内只有一个 #table', (html.match(/id="table"/g) || []).length === 1],
  ['导航仍是两列（.navcol + .navcol.icons）',
    /class="navcol"/.test(html) && /class="navcol icons"/.test(html)],
]) check(what, cond);

console.log(`\n合计：${pass} PASS / ${fail} FAIL`);
process.exit(fail ? 1 : 0);
