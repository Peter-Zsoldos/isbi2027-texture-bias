import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter
rng=np.random.default_rng(3); bw=25; rows=[]
for sig in [0.75,1.5]:
  for rep in range(6):
    x=gaussian_filter(rng.normal(size=(96,96,96)),sig,mode='wrap'); img=x/x.std()*60+40
    z,y,xx=np.ogrid[:96,:96,:96]
    roi=lambda c,r: (z-c[0])**2+(y-c[1])**2+(xx-c[2])**2<=r*r
    big=img[roi((48,48,48),40)]
    rmin=(np.floor(big/bw)-np.floor(big.min()/bw)+1).mean(); rfix=np.floor(big/bw).mean()
    for r in [3,4,5,6,8,10,14]:
        for t in range(10):
            c=rng.integers(r+1,95-r,3); v=img[roi(c,r)]
            rows.append(dict(sigma=sig,r=r,roi_min=(np.floor(v/bw)-np.floor(v.min()/bw)+1).mean()-rmin,fixed=np.floor(v/bw).mean()-rfix))
pd.DataFrame(rows).to_csv('anchoring.csv',index=False)
print(pd.DataFrame(rows).groupby(['sigma','r'])[['roi_min','fixed']].mean().round(2).unstack('sigma').to_string())
