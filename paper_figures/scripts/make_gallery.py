"""Local review gallery, not a manuscript figure; no external assets required."""
from pathlib import Path
from html import escape
ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper_figures/figures"
TITLES = {
    "fig01": ("逐指标模块贡献", "Three domains; each metric uses MSCA as the common denominator."),
    "fig02": ("扩散增量与用户重采样区间", "Mean relative gain over CoLiftRec; 95% paired-user bootstrap CI, not seed variation."),
    "fig03": ("两个模块的逐指标增量", "Panels use different baselines, with a shared numerical color scale."),
    "fig04": ("辅助损失权重扫描", "Baby reference-config sweep; † identifies R26_BASE, not M25."),
    "fig05": ("边际比例扫描", "Baby margin sweep; * identifies M25."),
    "fig06": ("角度约束扫描", "Baby reference-config sweep; † identifies R26_BASE, not M25."),
    "fig07": ("用户条件方向与成对优势", "Baby TRAIN diagnostics; selected epoch 23; medians and paired fractions, no confidence ribbon."),
    "fig08": ("候选分带中的方向与扰动", "Baby checkpoint replay: 118,551 examples per band; one-based rank labels."),
    "fig09": ("扰动覆盖率与上界触发率", "Same Baby replay; denominator includes examples with zero refinement."),
    "fig10": ("早停轨迹回放", "Completed epochs; retrospective analysis, not measured training time."),
    "fig11": ("三域扰动几何约束", "Mean and 90th percentile angles at selected checkpoints; not confidence intervals."),
    "fig12": ("三域验证集收敛", "Historical 60-epoch trajectories; unsmoothed, with selected checkpoints."),
    "fig13": ("内容相似与交互偏好诊断", "New CPU TRAIN leave-one-out probe; 2,000 Baby users; not Validation/Test metrics."),
    "fig14": ("新增 · 内容排名联合密度", "Paired Text/Visual ranks, marginal histograms, Top100 quadrants; 2,000 TRAIN targets."),
    "fig15": ("新增 · 偏好方向分布与雨云图", "Frozen Baby M25 CPU diagnostic: 4,096 paired TRAIN events per band; violin, median/IQR and paired DeltaA."),
    "fig16": ("新增 · 边际约束密度与角度分布", "Actual score shift versus original margin, plus empirical angle CDFs; no model update."),
    "fig17": ("新增 · Validation 边界排名流图", "Frozen B0/M25 Validation transitions; alluvial counts and unconditional per-band rescue/loss rates."),
    "fig18": ("新增 · CoLiftRec 模态消融", "Same frozen B0 checkpoint/candidates/backgrounds; Validation gains for T/A/V combinations."),
}

def main():
    cards=[]
    for role in ["main","supplementary"]:
        for p in sorted((FIG/role).glob("*.png")):
            key=p.stem.split("_")[0]
            title,note=TITLES[key]
            rel=p.relative_to(FIG).as_posix()
            stem=rel[:-4]
            recent = "true" if int(key[3:]) >= 14 else "false"
            cards.append(f'<article data-role="{role}" data-new="{recent}"><header><b>{key.upper()} · {escape(title)}</b><span>{role}</span></header>'
                         f'<a class="preview" href="{rel}" target="_blank"><img src="{rel}" alt="{escape(title)}"></a>'
                         f'<p>{escape(note)}</p><footer><a href="{stem}.pdf">PDF</a><a href="{stem}.svg">SVG</a><a href="{rel}">PNG</a></footer></article>')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Second-paper · 实验图稿</title><style>
body{font:15px/1.6 system-ui,sans-serif;color:#213047;background:#f5f7fa;margin:0;padding:36px;max-width:1440px;margin:auto}
h1{font-size:28px;margin:0} .intro{color:#617082;max-width:900px}nav{display:flex;gap:10px;margin:24px 0}
button{padding:8px 18px;border:1px solid #cad4df;background:white;border-radius:6px;cursor:pointer;color:#213047}button.active{background:#213047;color:white}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}article{background:white;border:1px solid #dce3eb;border-radius:8px;overflow:hidden}
header{display:flex;justify-content:space-between;align-items:center;padding:16px 20px;font-size:14px}header span{font-size:12px;color:#617082}
.preview{display:flex;align-items:center;justify-content:center;padding:10px;min-height:240px}img{max-width:100%;max-height:350px;object-fit:contain}
p{padding:0 20px;color:#617082;font-size:13px}footer{padding:12px 20px;border-top:1px solid #eef1f5;display:flex;gap:18px}a{color:#0072b2}
@media(max-width:850px){.grid{grid-template-columns:1fr}body{padding:18px}}[hidden]{display:none!important}
</style><h1>Second-paper · 实验图稿</h1><p class="intro">本轮新增 Fig14–18：联合密度、雨云分布、边际/角度分布、Validation 排名流、CoLift 模态消融。点击图片查看原尺寸；PDF/SVG 用于论文排版。实验正文优先考虑 Fig15、Fig17、Fig18，并按篇幅替换相应旧图。数据协议与英文图注见上级目录 VISUAL_REDESIGN_PLAN.md。</p>
<nav><button class="active" data-filter="new">本轮新增</button><button data-filter="all">全部</button><button data-filter="main">主图候选</button><button data-filter="supplementary">补充图</button></nav><section class="grid">'''+''.join(cards)+'''</section><script>
document.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('button').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('article').forEach(a=>a.hidden=b.dataset.filter==='new'?a.dataset.new!=='true':b.dataset.filter!=='all'&&a.dataset.role!==b.dataset.filter)}));
document.querySelector('button[data-filter="new"]').click();
</script></html>'''
    (FIG/"index.html").write_text(html,encoding="utf-8")
    print(f"Review gallery generated: {len(cards)} figures")
if __name__=="__main__": main()
