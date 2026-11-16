import ast
from pathlib import Path
import numpy as np
from simulation_gate import SimulationGate
for n in ('g1ra_backend.py','simulation_gate.py'): ast.parse(Path(__file__).with_name(n).read_text())
g=SimulationGate(); q=np.zeros(2); limits=np.array([[-1,1],[-1,1]])
assert not g.check(q,q,q,limits,.75,now=0)
for t in np.arange(.02,1.03,.02): ready=g.check(q,q,q,limits,.75,now=float(t))
assert ready
for a,b,c,h in [(q,q,q,.1),(q,q,np.array([2,0]),.75),(q,np.array([9,0]),q,.75),(np.array([float('nan'),0]),q,q,.75),(np.array([.4,0]),q,q,.75)]:
 assert not g.check(a,b,c,limits,h,now=1.04)
assert not g.check(q,q,q,limits,.75,now=2)
print('PASS: syntax, warmup, fallen, limits, speed, NaN, tracking error, interruption')
