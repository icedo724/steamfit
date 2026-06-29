"""training_frames.json → 학습과정 시각화 HTML(구조도 + 캔버스 애니메이션) 생성."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "viz" / "training_frames.json").read_text(encoding="utf-8"))
DATA_JS = json.dumps(DATA, ensure_ascii=False, separators=(",", ":"))

HTML = """<!DOCTYPE html><html lang="ko"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>SteamFit — 학습 과정 시각화</title>
<style>
:root{--deep:#03045e;--blue:#0077b6;--turq:#00b4d8;--frost:#90e0ef;--cyan:#caf0f8}
*{box-sizing:border-box}body{margin:0;background:var(--deep);color:var(--cyan);
font-family:-apple-system,BlinkMacSystemFont,'Noto Sans KR',sans-serif;line-height:1.5;padding:0 0 40px}
.wrap{max-width:560px;margin:0 auto;padding:0 16px}
header{padding:26px 16px 14px;background:linear-gradient(160deg,#0077b6,#03045e 80%)}
h1{font-size:1.4rem;margin:0 0 4px;color:#fff}.sub{color:var(--frost);font-size:.9rem}
h2{font-size:1.02rem;margin:26px 0 10px;color:#fff}
.card{background:var(--blue);border:1px solid var(--turq);border-radius:14px;padding:14px;margin-bottom:12px}
canvas{width:100%;height:340px;background:#021440;border-radius:10px;display:block}
.ctl{display:flex;align-items:center;gap:10px;margin-top:10px;flex-wrap:wrap}
button{background:var(--turq);color:var(--deep);border:none;border-radius:8px;padding:8px 16px;font-weight:700;cursor:pointer}
input[type=range]{flex:1;accent-color:var(--turq)}
.stat{font-size:.85rem;color:var(--frost)}.stat b{color:#fff}
.lgd{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;font-size:.72rem}
.lgd span{display:inline-flex;align-items:center;gap:4px;color:var(--cyan)}
.dot{width:9px;height:9px;border-radius:50%}
.arch{font-size:.84rem;color:var(--cyan)}
.flowrow{display:flex;align-items:center;gap:6px;flex-wrap:wrap;justify-content:center;margin:4px 0}
.node{background:#021440;border:1px solid var(--turq);border-radius:8px;padding:7px 10px;font-size:.78rem;text-align:center}
.node b{color:#fff}.ar{color:var(--turq);font-weight:700}
.note{font-size:.74rem;color:var(--frost);margin-top:8px}
small{color:var(--frost)}
</style></head><body>
<header><div class="wrap"><div class="sub">MODEL · 대조학습 임베딩</div>
<h1>학습 과정 시각화</h1>
<div class="sub">게임 임베딩이 학습되며 군집을 형성하는 과정 (TF-Playground 스타일)</div></div></header>
<div class="wrap">

<h2>🎬 학습 애니메이션</h2>
<div class="card">
  <canvas id="cv" width="700" height="340"></canvas>
  <div class="ctl">
    <button id="play">⏸ 일시정지</button>
    <button id="replay">↻ 처음부터</button>
    <input type="range" id="sld" min="0" max="1" step="0.01" value="0">
  </div>
  <div class="ctl"><span class="stat">에폭 <b id="ep">0</b> / __EPOCHS__ · 대조손실 <b id="loss">—</b></span></div>
  <div class="lgd" id="lgd"></div>
  <div class="note">처음엔 무작위로 흩어진 점(게임)들이, 대조학습이 진행될수록 <b>같이 플레이되는 게임끼리 가까이</b> 모입니다. 점 = 게임, 색 = 장르.</div>
</div>

<h2>🧩 모델 구조 (대조학습)</h2>
<div class="card arch">
  <div class="flowrow">
    <div class="node">게임 A<br><b>(앵커)</b></div>
    <div class="node">게임 B<br><b>(양성쌍)</b></div>
  </div>
  <div class="flowrow"><span class="ar">↓ 임베딩 룩업</span></div>
  <div class="flowrow"><div class="node"><b>임베딩 테이블</b><br>게임 N × 차원 D</div></div>
  <div class="flowrow"><span class="ar">↓ L2 정규화</span></div>
  <div class="flowrow"><div class="node"><b>유사도 행렬</b><br>배치 내 모든 쌍 (B×B)</div></div>
  <div class="flowrow"><span class="ar">↓ InfoNCE 손실</span></div>
  <div class="flowrow"><div class="node"><b>양성쌍은 가깝게</b> · <b>나머지(배치 내 음성)는 멀게</b></div></div>
  <div class="note">· 협업 임베딩(item2vec식): 게임 ID → 벡터, 공동플레이 쌍으로 학습 (위 그림)<br>
  · 콘텐츠 임베딩: 텍스트(이름·태그·설명) → 다국어 트랜스포머 → 벡터<br>
  · 추천 = 협업(취향) + 콘텐츠(의도) 가중 결합 → 하이브리드</div>
</div>
<small>* 애니메이션은 상위 __NITEMS__개 게임으로 학습한 실제 궤적(에폭별 임베딩을 최종 PCA 평면에 투영). 실제 배포 모델은 1.2만 게임·256차원.</small>
</div>
<script>
const D=__DATA__;
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
function rs(){const r=cv.getBoundingClientRect();cv.width=r.width*2;cv.height=r.height*2;ctx.setTransform(2,0,0,2,0,0);}
rs();window.addEventListener('resize',rs);
const b=D.bounds,pad=16;
function W(){return cv.width/2}function H(){return cv.height/2}
function px(x,y){return [pad+(x-b.xlo)/(b.xhi-b.xlo)*(W()-2*pad), H()-pad-(y-b.ylo)/(b.yhi-b.ylo)*(H()-2*pad)];}
const pal=['#00b4d8','#90e0ef','#caf0f8','#48cae4','#ade8f4','#0096c7','#0077b6','#48cae4','#90e0ef','#caf0f8','#00b4d8'];
const gs=[...new Set(D.genres)],gc={};gs.forEach((g,i)=>gc[g]=pal[i%pal.length]);
document.getElementById('lgd').innerHTML=gs.map(g=>`<span><span class="dot" style="background:${gc[g]}"></span>${g}</span>`).join('');
const NF=D.frames.length;document.getElementById('sld').max=NF-1;
let t=0,playing=true,speed=0.035;
function interp(t){const f0=Math.floor(t),f1=Math.min(f0+1,NF-1),a=t-f0,A=D.frames[f0],B=D.frames[f1];
  return A.map((p,i)=>[p[0]+(B[i][0]-p[0])*a,p[1]+(B[i][1]-p[1])*a]);}
function draw(){ctx.clearRect(0,0,W(),H());const pts=interp(t);
  for(let i=0;i<pts.length;i++){const q=px(pts[i][0],pts[i][1]);ctx.fillStyle=gc[D.genres[i]];ctx.globalAlpha=.8;ctx.beginPath();ctx.arc(q[0],q[1],2.4,0,7);ctx.fill();}
  ctx.globalAlpha=1;const ep=Math.round(t);
  document.getElementById('ep').textContent=ep;
  document.getElementById('loss').textContent=ep>0?D.losses[Math.min(ep-1,D.losses.length-1)]:'—';
  document.getElementById('sld').value=t;}
function loop(){if(playing){t+=speed;if(t>=NF-1){t=NF-1;playing=false;document.getElementById('play').textContent='▶ 재생';}}draw();requestAnimationFrame(loop);}
document.getElementById('play').onclick=()=>{if(t>=NF-1)t=0;playing=!playing;document.getElementById('play').textContent=playing?'⏸ 일시정지':'▶ 재생';};
document.getElementById('replay').onclick=()=>{t=0;playing=true;document.getElementById('play').textContent='⏸ 일시정지';};
document.getElementById('sld').oninput=e=>{t=parseFloat(e.target.value);playing=false;document.getElementById('play').textContent='▶ 재생';draw();};
loop();
</script></body></html>"""

html = (HTML.replace("__DATA__", DATA_JS)
            .replace("__EPOCHS__", str(DATA["epochs"]))
            .replace("__NITEMS__", str(DATA["n_items"])))
(ROOT / "viz" / "training_viz.html").write_text(html, encoding="utf-8")
print("저장: viz/training_viz.html  (%.0f KB)" % (len(html) / 1024))
