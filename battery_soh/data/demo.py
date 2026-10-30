"""Synthetic data solely for smoke testing the pipeline. NOT scientific results."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .schema import harmonize

def synthetic_datasets(seed=73, batteries=7, observations=36):
    rng=np.random.default_rng(seed)
    rows=[]
    for ds,chem,nominal,slope in [('toy_a','LCO',2.,.24),('toy_b','LFP',1.1,.18),('toy_c','NCM',2.,.31)]:
        for b in range(batteries):
            bias=rng.normal(0,1.0)
            speed=slope*rng.uniform(.75,1.25)
            for cyc in range(1,observations+1):
                soh=98+bias-speed*cyc+rng.normal(0,.35)
                rows.append({'dataset':ds,'battery_id':f'{ds}_cell{b:02d}',
                             'cycle':cyc,'rated_capacity_ah':nominal,
                             'capacity_ah':nominal*soh/100,'chemistry':chem})
    return harmonize(pd.DataFrame(rows))
