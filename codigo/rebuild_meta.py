"""Restore the full metadata columns of the duplicate maps from your own copy of CP-AnemiC version 1.

The public maps (codigo/meta_groups*.csv, gpu/meta_groups_geo.csv) keep only image_id, image_path, anaemia label, SHA-256
digest and group id: Hb, age, sex and facility of the 710 children are not redistributed, because the dataset was withdrawn
by its authors without a stated reason. The analysis scripts that need those columns (Hb regression, metadata discordance,
leave-one-hospital-out, WHO-2024 relabelling) read them from the maps, so run this once on a local copy before running them:

    python codigo/rebuild_meta.py <folder with the original metadata.csv and images/>

The script checks the SHA-256 digest of every image against the map before writing anything, and rewrites the maps in place
in YOUR working copy. Do not commit the rebuilt files.
"""
import sys, os, hashlib, pandas as pd
src = sys.argv[1]
meta = pd.read_csv(os.path.join(src, 'metadata.csv'))
need = ['image_id', 'hb', 'severity', 'hospital', 'age_months', 'gender']
missing = [c for c in need if c not in meta.columns]
if missing: sys.exit('metadata.csv lacks columns %s (expected the ingested CP-AnemiC v1 layout)' % missing)
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here)
for rel in ('codigo/meta_groups.csv', 'codigo/meta_groups_strict.csv', 'codigo/meta_groups_geo.csv', 'gpu/meta_groups_geo.csv'):
    path = os.path.join(root, rel); m = pd.read_csv(path)
    if 'hb' in m.columns: print('already full:', rel); continue
    bad = [r.image_id for r in m.itertuples() if hashlib.sha256(open(os.path.join(src, r.image_path), 'rb').read()).hexdigest() != r.sha]
    if bad: sys.exit('%d images differ from the analysed copy (first: %s); not writing %s' % (len(bad), bad[0], rel))
    full = m.merge(meta[need], on='image_id', how='left', validate='one_to_one')
    full = full[['image_id', 'image_path', 'hb', 'anemia', 'severity', 'hospital', 'age_months', 'gender', 'sha', 'grp']]
    full.to_csv(path, index=False); print('restored', rel)
