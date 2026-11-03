"""Read-only PhysicalNet material audit; all new records stay in this project."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import csv,hashlib,json,re
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT.parent/'PhysicalNet'
OUT=ROOT/'outputs/analysis/material_dataset_audit_20261020'
OUT.mkdir(parents=True,exist_ok=True)
DATASETS={
 'WS2_public':('other data/WS2_data',['BG','1L','FL','ML'],'repository describes public; original citation/license not located'),
 'Graphene_public':('other data/graphene',['BG','1L','>1L'],'repository describes public; original citation/license not located'),
 'MoS2_external':('other_datav2_prepared',['BG','1L','FL','ML'],'repository calls external public; original label mapping described as assumption'),
 'WS2_supp':('supplementary_prepared/WS2_supp',['BG','1L','FL','ML'],'filtered internal supplementary v2 per historical docs'),
 'Graphene_supp':('supplementary_prepared/Gr_supp',['BG','FL','ML'],'filtered internal supplementary v2; differs from public graphene'),
 'WS2_combined':('supplementary_prepared/WS2_combined',['BG','1L','FL','ML'],'derived union of public train and supplementary train; public val/test'),
}
EXT={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
records=[];datasets={};source_hashes={};issues=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as fp:
  for block in iter(lambda:fp.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def touch(p):
 source_hashes[str(p)]=sha(p);return source_hashes[str(p)]
def scan_pair(dataset,split,image,mask,k):
 try:
  with Image.open(image) as im:
   mode=im.mode;rgb=np.asarray(im.convert('RGB'));size=im.size
   thumb=im.convert('L').resize((9,8),Image.Resampling.BILINEAR)
  with Image.open(mask) as im:
   m=np.asarray(im);mask_mode=im.mode;mask_size=im.size
  vals,cnt=np.unique(m,return_counts=True)
  pixel_counts={str(int(v)):int(c) for v,c in zip(vals,cnt)}
  err=[]
  if m.ndim!=2:err.append('mask_not_single_index_channel')
  if size!=mask_size:err.append('image_mask_size_mismatch')
  if any(int(v)<0 or int(v)>=k for v in vals):err.append('unexpected_mask_values')
  pix_sha=hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()
  a=np.asarray(thumb);bits=(a[:,1:]>a[:,:-1]).ravel();dhash=sum(int(v)<<i for i,v in enumerate(bits))
  parent_match=re.match(r'^(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}\.\d+)_\d+$',image.stem)
  rec={'dataset':dataset,'split':split,'id':image.stem,'image':str(image),'mask':str(mask),
       'width':size[0],'height':size[1],'image_mode':mode,'mask_mode':mask_mode,
       'image_sha256':touch(image),'mask_sha256':touch(mask),'decoded_rgb_sha256':pix_sha,
       'mask_values':[int(v) for v in vals],'pixel_counts':pixel_counts,
       'dhash64':f'{dhash:016x}','timestamp_parent_candidate':parent_match[1] if parent_match else None,'errors':err}
  if err:issues.append({'dataset':dataset,'split':split,'id':image.stem,'errors':err})
  return rec
 except Exception as exc:
  issues.append({'dataset':dataset,'split':split,'id':image.stem,'errors':[repr(exc)]})
  return None

for name,(relative,classes,provenance) in DATASETS.items():
 directory=SRC/relative;item={'root':str(directory),'classes_from_docs':classes,'provenance_status':provenance,'splits':{}}
 for split in ['train','val','test']:
  idir=directory/'img_dir'/split;adir=directory/'ann_dir'/split
  images=sorted(p for p in idir.iterdir() if p.is_file() and p.suffix.lower() in EXT) if idir.exists() else []
  masks=sorted(p for p in adir.iterdir() if p.is_file() and p.suffix.lower() in EXT) if adir.exists() else []
  imap=defaultdict(list);amap=defaultdict(list)
  for p in images:imap[p.stem].append(p)
  for p in masks:amap[p.stem].append(p)
  sample=[]
  for identity,paths in sorted(imap.items()):
   if identity in amap and len(paths)==len(amap[identity])==1:
    r=scan_pair(name,split,paths[0],amap[identity][0],len(classes))
    if r:records.append(r);sample.append(r)
  pixels=Counter();present=Counter()
  for r in sample:
   pixels.update(r['pixel_counts']);present.update(str(v) for v in r['mask_values'])
  count=sum(pixels.values())
  item['splits'][split]={'directory_exists':idir.is_dir(),'images':len(images),'masks':len(masks),'valid_pairs':len(sample),
      'missing_masks':sorted(set(imap)-set(amap)),'orphan_masks':sorted(set(amap)-set(imap)),
      'duplicate_image_stems':[x for x,v in imap.items() if len(v)>1],
      'duplicate_mask_stems':[x for x,v in amap.items() if len(v)>1],
      'sizes':dict(Counter(f"{r['width']}x{r['height']}" for r in sample)),
      'min_dimension':min((min(r['width'],r['height']) for r in sample),default=None),
      'pixel_counts':dict(pixels),'pixel_percent':{c:100*n/count for c,n in pixels.items()} if count else {},
      'images_containing_class':dict(present),'mask_modes':dict(Counter(r['mask_mode'] for r in sample))}
  txt=directory/(split+'.txt')
  if txt.exists():
   ids=[x.strip() for x in txt.read_text().splitlines() if x.strip()];touch(txt)
   item['splits'][split]['text_manifest_count']=len(ids)
   item['splits'][split]['text_manifest_vs_actual_symmetric_difference']=sorted(set(ids)^set(imap))
 stat=directory/'stats_train.json'
 if stat.exists():touch(stat);item['existing_train_stats_cache']=json.loads(stat.read_text())
 extra=directory/'ann_dir/test_label'
 if extra.exists():
  item['extra_test_label_directory']={'files':len([p for p in extra.iterdir() if p.is_file()]),'note':'auxiliary annotation folder, not an extra image/test split'}
 datasets[name]=item
 print('SCANNED',name,{s:r['valid_pairs'] for s,r in item['splits'].items()},flush=True)

# Main MoS2: only decode RGB for duplicate checks; its full mask audit is already frozen.
main=[]
for split in ['train','val','test']:
 p=SRC/'splits'/f'{split}.txt';touch(p)
 for identity in p.read_text().split():
  path=SRC/'Mos2_data/ori/MoS2'/f'{identity}.jpg';digest=touch(path)
  with Image.open(path) as im:rgb=np.asarray(im.convert('RGB'))
  main.append({'dataset':'MoS2_main','split':split,'id':identity,'image':str(path),
      'image_sha256':digest,'decoded_rgb_sha256':hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()})

all_records=records+main
groups=defaultdict(list)
for r in all_records:groups[r['decoded_rgb_sha256']].append({k:r[k] for k in ['dataset','split','id','image']})
duplicates=[v for v in groups.values() if len(v)>1]
within_cross_split=[]
for group in duplicates:
 for dataset in set(r['dataset'] for r in group):
  subset=[r for r in group if r['dataset']==dataset]
  if len(set(r['split'] for r in subset))>1:within_cross_split.append(subset)
parents=[]
for dataset in DATASETS:
 byparent=defaultdict(list)
 for r in records:
  if r['dataset']==dataset and r['timestamp_parent_candidate']:
   byparent[r['timestamp_parent_candidate']].append({'split':r['split'],'id':r['id']})
 crossed=[{'prefix':key,'samples':group} for key,group in byparent.items() if len(set(x['split'] for x in group))>1]
 parents.append({'dataset':dataset,'recognized_timestamp_prefixes':len(byparent),
     'cross_split_prefix_count':len(crossed),'cross_split_prefixes':crossed,
     'interpretation':'Filename evidence suggests shared acquisition/parent; source trace needed, not proof of crop geometry.'})

near=[]
for i,a in enumerate(records):
 for b in records[i+1:]:
  if a['dataset']!=b['dataset'] or a['split']==b['split']:continue
  if a['decoded_rgb_sha256']==b['decoded_rgb_sha256']:continue
  if abs(a['width']/a['height']-b['width']/b['height'])>0.02:continue
  distance=(int(a['dhash64'],16)^int(b['dhash64'],16)).bit_count()
  if distance<=2:near.append({'dataset':a['dataset'],'a':a['image'],'a_split':a['split'],
     'b':b['image'],'b_split':b['split'],'dhash_hamming':distance,
     'interpretation':'Candidate similarity only; no automatic deletion or split reassignment.'})

overlap_matrix=[]
for i,a in enumerate(DATASETS):
 sa={r['decoded_rgb_sha256'] for r in records if r['dataset']==a}
 for b in list(DATASETS)[i+1:]:
  sb={r['decoded_rgb_sha256'] for r in records if r['dataset']==b}
  if sa&sb:overlap_matrix.append({'dataset_a':a,'dataset_b':b,'shared_unique_rgb_images':len(sa&sb)})

docs=['DATA.md','README.md','compare_refer/docs/transfer_three_line_design.md',
      'compare_refer/docs/transfer_results.md','compare_refer/other_datav2/data_audit.md',
      'transfer_train/README.md','scripts/prepare_supplementary_splits.py',
      'scripts/prepare_other_datav2_split.py','transfer/material_dataset.py','transfer/finetune.py']
for relative in docs:
 p=SRC/relative
 if p.is_file():touch(p)

frozen=json.loads((ROOT/'outputs/launches/p1_reference_adapter_20261018/plan.json').read_text())
guard_checks={}
for key in ['source_sha256','protected_artifacts_sha256','workflow_sha256','preflight_artifacts_sha256']:
 bad=[]
 for relative,expected in frozen[key].items():
  p=Path(relative);p=p if p.is_absolute() else ROOT/p
  if sha(p)!=expected:bad.append(str(p))
 guard_checks[key]={'count':len(frozen[key]),'mismatches':bad}

# Verify the original files remained byte-identical during this audit.
changed=[p for p,expected in source_hashes.items() if sha(Path(p))!=expected]
assert not changed,changed
audit={'status':'completed','created_at_beijing':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
  'scope':'Read-only data/content audit and planning; no training, optimizer, model forward, cache writes or test metrics.',
  'datasets':datasets,'samples':records,'main_mos2_duplicate_check_samples':len(main),
  'pairing_or_label_errors':issues,'within_dataset_exact_cross_split_duplicates':within_cross_split,
  'decoded_rgb_duplicate_groups_across_all_roots':duplicates,'dataset_overlap_counts':overlap_matrix,
  'timestamp_parent_cross_split_checks':parents,'near_duplicate_candidates':near,
  'source_files_sha256':source_hashes,'source_files_changed_during_audit':changed,
  'conditional_project_frozen_checks':guard_checks,
  'limits':['No source/parent/sample independence is inferred from hash inequality.',
            'Numeric IDs do not prove label semantics; provenance docs still require original-source confirmation.',
            'Repository claims public, but original publication/DOI/license not found in inspected notices.',
            'Historical PhysicalNet test/val metrics and selection already exist; no new blind-set claim.']}
(OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
(OUT/'samples.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
with (OUT/'dataset_summary.csv').open('w',newline='',encoding='utf-8-sig') as fp:
 w=csv.writer(fp);w.writerow(['dataset','root','classes','train','val','test','train_sizes','val_sizes','test_sizes','train_class_pixel_percent'])
 for dataset,d in datasets.items():
  w.writerow([dataset,d['root'],'/'.join(d['classes_from_docs']),*[d['splits'][s]['valid_pairs'] for s in ['train','val','test']],
       *[json.dumps(d['splits'][s]['sizes']) for s in ['train','val','test']],json.dumps(d['splits']['train']['pixel_percent'])])
print(json.dumps({'datasets':{n:{s:d['splits'][s]['valid_pairs'] for s in ['train','val','test']} for n,d in datasets.items()},
 'pairing_errors':len(issues),'within_dataset_exact_cross_split_duplicates':len(within_cross_split),
 'overlaps':overlap_matrix,'parent_prefix_cross_split':{d['dataset']:d['cross_split_prefix_count'] for d in parents},
 'near_duplicate_candidates':len(near),'unchanged_source_files':len(source_hashes),'frozen_checks':guard_checks},ensure_ascii=False,indent=2))
