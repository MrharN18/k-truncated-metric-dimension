import json, sys
from tadpole_core import TadpoleGraph, is_k_truncated_resolving
from tadpole_exact import solve_for_config
from pendant_placement import global_configurations
cases=json.load(open(sys.argv[1])); out=[]
for idx,c in enumerate(cases):
 g=TadpoleGraph(c['n'], [tuple(x) for x in c['paths']], c['k']); d=g.all_pairs_dist()
 best=None; bestset=None
 for gaps in global_configurations([m for _,m in g.pendant_paths],g.k):
  sp,sc=solve_for_config(g,d,gaps); S=sp|sc
  if is_k_truncated_resolving(g,S,d) and (best is None or len(S)<best): best=len(S); bestset=sorted(S)
 out.append({'i':idx,'alg':best,'set':bestset})
json.dump(out,sys.stdout)
