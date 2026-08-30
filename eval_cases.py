import json, sys
from tadpole_core import TadpoleGraph, is_k_truncated_resolving, brute_force_min_resolving
from tadpole_exact import solve_for_config
from pendant_placement import global_configurations

cases=json.load(open(sys.argv[1]))
out=[]
for idx,c in enumerate(cases):
    g=TadpoleGraph(c['n'], [tuple(x) for x in c['paths']], c['k'])
    d=g.all_pairs_dist()
    best=None; bestset=None; valid_configs=0
    for gaps in global_configurations([m for _,m in g.pendant_paths], g.k):
        sp,sc=solve_for_config(g,d,gaps)
        S=sp|sc
        if is_k_truncated_resolving(g,S,d):
            valid_configs+=1
            if best is None or len(S)<best:
                best=len(S); bestset=sorted(S)
    brute=None
    if len(g.all_vertices)<=15:
        b=brute_force_min_resolving(g,dist=d)
        brute=len(b) if b is not None else None
    out.append({'i':idx,'alg':best,'set':bestset,'brute':brute,'valid_configs':valid_configs,'V':len(g.all_vertices)})
json.dump(out,sys.stdout)
