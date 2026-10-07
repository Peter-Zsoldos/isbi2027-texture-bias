import numpy as np, SimpleITK as sitk, logging, time
from scipy.ndimage import gaussian_filter
from radiomics import featureextractor
logging.getLogger('radiomics').setLevel(logging.ERROR)
FEATS={'firstorder':['Entropy','Uniformity'],'glcm':['JointEntropy','Contrast','Idm','Correlation'],'glrlm':['RunEntropy'],'glszm':['ZoneEntropy']}
ext=featureextractor.RadiomicsFeatureExtractor(binWidth=25, force2D=False)
ext.disableAllFeatures()
for k,v in FEATS.items(): ext.enableFeaturesByName(**{k:v})
rng=np.random.default_rng(0)
def texture(n=96):
    x=gaussian_filter(rng.normal(size=(n,n,n)),1.5); return (x/x.std()*60+40).astype(np.float32)   # 'HU'-like
def feats(img,mask):
    I=sitk.GetImageFromArray(img); M=sitk.GetImageFromArray(mask.astype(np.uint8))
    r=ext.execute(I,M); return {k.split('_',2)[-1]:float(v) for k,v in r.items() if k.startswith('original_')}
def sphere(n,c,r):
    z,y,x=np.ogrid[:n,:n,:n]; return (z-c[0])**2+(y-c[1])**2+(x-c[2])**2<=r*r
def jackknife(img,mask,c):
    th=feats(img,mask); z,y,x=np.nonzero(mask)
    octs=[((z>=c[0])==a)&((y>=c[1])==b)&((x>=c[2])==d) for a in (0,1) for b in (0,1) for d in (0,1)]
    loo=[]
    for o in octs:
        m2=mask.copy(); m2[z[o],y[o],x[o]]=False; loo.append(feats(img,m2))
    B=8; return th,{k:B*th[k]-(B-1)*np.mean([l[k] for l in loo]) for k in th}
t0=time.time(); res=[]
for rep in range(3):
    img=texture(); n=img.shape[0]; c=(n//2,)*3
    ref=feats(img,sphere(n,c,40))
    for r in [3,4,6,9,14]:
        cc=tuple(rng.integers(20,n-20,3))
        th,jk=jackknife(img,sphere(n,cc,r),cc)
        res.append((r,ref,th,jk))
print('time',round(time.time()-t0))
keys=list(res[0][1].keys())
for k in keys:
    line=[]
    for r in [3,4,6,9,14]:
        rr=[x for x in res if x[0]==r]
        b=np.mean([x[2][k]-x[1][k] for x in rr]); j=np.mean([x[3][k]-x[1][k] for x in rr])
        line.append(f"r{r}: {b:+.3f}/{j:+.3f}")
    sd=np.std([x[1][k] for x in res])
    print(f"{k:14s} ref={np.mean([x[1][k] for x in res]):.3f} | plug-in/jackknife bias: "+'  '.join(line))
