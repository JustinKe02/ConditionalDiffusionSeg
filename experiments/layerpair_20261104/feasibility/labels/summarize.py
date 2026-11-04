import json,statistics
from pathlib import Path
d=json.loads((Path(__file__).parent/'components.json').read_text());print('dup',d['dup'])
def ratio(a,b):return str(a)+'/'+str(b)+' ('+format(a/max(1,b),'.1%')+')'
for drop in (None,'m169','m56'):
 data={i:{int(c):comp for c,comp in row.items()} for i,row in d['stats'].items() if i!=drop}
 print('Exclude',drop, 'n',len(data))
 for idx,threshold in ((0,1),(1,1),(2,1),(3,1)):
  frames={}
  for i in data:
   frames[i]={c:[a for a in data[i][c] if a[idx]>=threshold] for c in (1,2)}
  nt=0
  for row in frames.values():
   n1,n2=[len(row[c]) for c in (1,2)]
   nt+=n1*n2
  print('qualifying idx',idx,'image n1,n2,both,multi,thin',*[sum(int(q(r)) for r in frames.values()) for q in (lambda r:bool(r[1]),lambda r:bool(r[2]),lambda r:bool(r[1] and r[2]),lambda r:len(r[1])+len(r[2])>=2,lambda r:bool(r[1] or r[2]))],'distinct-pair',nt)
  if idx!=2 or drop=='m56':continue
  for c in (1,2):
   pix=sum(a[0] for r in data.values() for a in r[c]);regs=sum(len(r[c]) for r in data.values())
   print('class',c,'valid relative nums/pix',ratio(sum(len(r[c]) for r in frames.values()),regs),ratio(sum(a[0] for r in frames.values() for a in r[c]),pix))
   for condition in ("any", "opp"):
    n=p=0
    for i,r in data.items():
     valid=frames[i]
     for a in r[c]:
      can=bool(valid[3-c]) if condition=='opp' else len(valid[1])+len(valid[2])-(a[2]>=1)>0
      n+=can;p+=int(can)*a[0]
    print(condition,'ref query comps/pix',ratio(n,regs),ratio(p,pix))
