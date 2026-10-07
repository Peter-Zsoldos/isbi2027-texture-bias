import numpy as np, SimpleITK as sitk, logging, time
from scipy.ndimage import gaussian_filter
from radiomics import featureextractor
logging.getLogger('radiomics').setLevel(logging.ERROR)
FEATS={'firstorder':['Entropy','Uniformity'],'glcm':['JointEntropy','Contrast','Correlation'],'glrlm':['RunEntropy']}
ext=featureextractor.RadiomicsFeatureExtractor(binWidth=25); ext.disableAllFeatures()
for k,v in FEATS.items(): ext.enableFeaturesByName(**{k:v})
rng=np.random.default_rng(1)
def feats(img,mask):
    r=ext.execute(sitk.GetImageFromArray(img),sitk.GetImageFromArray(mask.astype(np.uint8)))
    return np.array([float(v) for k,v in r.items() if k.startswith('original_')])
names=[f'{a}' for k,v in FEATS.items() for a in v]
def sphere(n,c,r):
    z,y,x=np.ogrid[:n,:n,:n]; return (z-c[0])**2+(y-c[1])**2+(x-c[2])**2<=r*r
def corrected(img,mask,c):
    z,y,x=np.nonzero(mask); N=len(z)
    oct_id=(z>=c[0])*4+(y>=c[1])*2+(x>=c[2])
    full=feats(img,mask)
    loo=[]; halves=[]
    for o in range(8):
        m=mask.copy(); s=oct_id==o; m[z[s],y[s],x[s]]=False; loo.append(feats(img,m))
    for ax,(arr,cc) in enumerate([(z,c[0]),(y,c[1]),(x,c[2])]):
        for side in (0,1):
            m=mask.copy(); s=(arr>=cc)==side; m[z[s],y[s],x[s]]=False; halves.append(feats(img,m))
    loo=np.mean(loo,0); halves=np.mean(halves,0)
    jk=8*full-7*loo
    # fit theta(n)=a+b/n+c/n^2 through n=N, 7N/8, N/2 (exact)  -> in units u=N/n: u=1,8/7,2
    U=np.array([1,8/7,2.0]); A=np.stack([np.ones(3),U,U**2],1)
    quad=np.linalg.solve(A,np.stack([full,loo,halves]))[0]
    lin=np.linalg.lstsq(A[:,:2],np.stack([full,loo,halves]),rcond=None)[0][0]
    return full,jk,lin,quad
t0=time.time(); out={r:[] for r in [3,4,6,9]}
for rep in range(8):
    x=gaussian_filter(rng.normal(size=(112,112,112)),1.5); img=(x/x.std()*60+40).astype(np.float32); n=112
    ref=feats(img,sphere(n,(56,56,56),48))
    for r in out:
        for t in range(2):
            cc=tuple(rng.integers(16,n-16,3)); out[r].append(np.stack(corrected(img,sphere(n,cc,r),cc))-ref)
print('time',round(time.time()-t0))
print('bias: plug-in / jackknife / lin-extrap / quad-extrap   (sd of plug-in error)')
for i,nm in enumerate(names):
    print(f"{nm:13s}",'  '.join(f"r{r}: "+'/'.join(f"{v:+.2f}" for v in np.mean(out[r],0)[:,i])+f" ({np.std(np.array(out[r])[:,0,i]):.2f})" for r in out))
