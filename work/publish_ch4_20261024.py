import json,hashlib,zipfile,shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path('/root/autodl-tmp/ConditionalDiffusionSeg');D=R/'docs/figures/chapter4_20261024';P=R/'docs/thesis/chapter4_20261024'
archive=R/'work/chapter4_completion_20261024.zip'
D.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(archive) as z:
 for entry in z.infolist():
  parts=Path(entry.filename).parts[1:]
  assert parts and '..' not in parts and not Path(*parts).is_absolute()
  out=D.joinpath(*parts)
  if entry.is_dir():out.mkdir(parents=True,exist_ok=True)
  else:
   data=z.read(entry)
   if out.exists():assert out.read_bytes()==data, f'Preserve differing existing delivery file: {out}'
   else:out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
manifest=json.loads((D/'file_manifest.json').read_text())
for rel,digest in manifest.items():assert hashlib.sha256((D/rel.replace('\\','/')).read_bytes()).hexdigest()==digest,rel
P.mkdir(parents=True,exist_ok=False)
for name in ['chapter4_manuscript.docx','chapter4_manuscript.pdf','chapter4_manuscript.txt','abstract_and_structure.txt','delivery_checks.json','sources.json']:
 shutil.copy2(D/name,P/name)
note='''

## 65. 第四章扩展批次剩余测速与图表正文完成 2026-10-24

用户在本对话明确授权“开始执行剩余的计划”。复用已完成的E3R导出、RepELA工程验证/100轮训练/test及既有6项测速，未重训、未重跑test，未改PhysicalNet、原数据/划分、冻结源文件和既有权重。

独立新增work/complete_ch4_20261024.py修复FlopCounterMode模块梯度跟踪在no_grad参数输入上的AssertionError：保留原PyTorch ATen算子计数注册表，仅统计Global总量并关闭不需要的模块/autograd hook；Conv、bmm解析计数及LayerNorm参数输入检查通过。旧冻结计划/代码未修改。旧failed状态保存在outputs/launches/p1_ch4_extension_20261024/completion_revision_20261024/status_before.json，修订来源和runner哈希见revision.json。17/17测速于2026-10-24 12:05:42北京时间完成，原生算子一致率门槛均通过，原始冻结文件、PhysicalNet源/副本、12个既有best权重哈希前后校验通过。本批status.json=completed，最终comparison_summary.json/final_analysis.md已生成。

RTX3090 FP32统一测速：E3R/RepELA部署形态参数18.197/2.123M，单窗198.232/5.249GMACs，单窗134.60/20.81ms，整图预测7.112/3.533s；E3R/RepELA单窗比6.47，整图比2.01。新原生算子口径不得混用旧确定性评估时间。部署形态不是比训练形态各指标必然更快，如实保留实际计时。

阶段5及新版正文：docs/figures/chapter4_20261024/为完整交付（17张图各PDF/SVG/PNG与sources，7项正文表+2项附表分11个CSV，机制统计、来源哈希、脚本、README.txt）。docs/thesis/chapter4_20261024/包含17页Word/PDF新稿、正文文本、摘要/贡献/目录衔接建议和核验。Word原生公式；默认LibreOffice渲染器不可用后用Microsoft Word导出PDF，逐页PNG检查；图表边框闭合、标题黑色、无默认蓝线、表格无越界。绘图Tinos/NotoSerifSC，个别缺字SimSun补足；Word宋体/Times New Roman。旧20261019初稿保持不变。

过程选图m22/m132按候选中位数规则；定性m149/m80按GT薄层组成选取。改动图ROI按无符号边界附近改动密度选，不按改对/改错；边界概率点按无符号总变化展示动态并明示非典型性估计。逐步改动计数按全部25张GT/端点CPU汇总，与原report比例一致。旧§62中的2.4/1.8/1.7‰换算大约10倍，正确为0.2364/0.1760/0.1725‰。边界置信度均值0.8746/0.8725/0.8773/0.8788，不是单调收敛；正文已修正。E3R区域指标低于SegFormer/E0及原筛选未通过事实保留。

仍需外部资料：样品/母图来源独立性和物理标签依据、学校模板、真实边缘设备及对应部署资料；不声称本批完成新边缘部署，不自动追加训练/种子/跨材料/测试。后续从新稿与README.txt继续。
'''
with (R/'HANDOFF.md').open('a',encoding='utf-8') as f:f.write(note)
result={'published_bjt':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'figures_path':str(D),'manuscript_path':str(P),'verified_files':len(manifest),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'handoff_section':65}
(R/'outputs/launches/p1_ch4_extension_20261024/paper_delivery_20261024.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False),flush=True)
