# Run make_figs_b.py section by section so that figures whose inputs are not ready yet are skipped (not crash the rest).
import re, sys, traceback
src=open('/home/claude/figs_new/make_figs_b.py').read()
parts=re.split(r'(?m)^(?=# ---------------- )',src); ns={'__name__':'__main__'}
exec(compile(parts[0],'header','exec'),ns)
only=sys.argv[1:]  # optional: substrings of section titles to run
for p in parts[1:]:
    title=p.split('\n',1)[0]
    if only and not any(o.lower() in title.lower() for o in only): continue
    try: exec(compile(p,title,'exec'),ns); print('OK  ',title)
    except Exception as e: print('SKIP',title,'->',type(e).__name__,str(e)[:120])
