import sys, subprocess, os
sys.path.insert(0, '/mnt/workspace/node1/dream/tools')
import dream_loop as dl

m, t = dl.build_model('/mnt/workspace/node1/models/Qwen2.5-0.5B-Instruct', None)
idxs = [0, 1, 2, 5, 6, 7, 8, 12]
for i in idxs:
    s = dl.DREAM_BANK[i]
    p = dl.build_prompt(s['task'], s.get('hint', ''), [])
    g = dl.gen_code(m, t, p, max_new=512, temp=0.0)
    lang, code = dl.extract_code(g)
    print("### task[%d] %s  len=%d" % (i, s['task'][:46], len(code or '')))
    if not code:
        print("  NO CODE"); continue
    fn = '/tmp/dd_%d.py' % i
    open(fn, 'w').write(code)
    r = subprocess.run([sys.executable, fn], capture_output=True, text=True,
                       timeout=90, cwd='/tmp', env={**os.environ, 'MPLBACKEND': 'Agg'})
    if r.returncode:
        print("  rc=1  TRACEBACK_TAIL:")
        print("   " + "\n   ".join(r.stderr.strip().splitlines()[-4:]))
    else:
        print("  rc=0 OK mp4=", os.path.exists('/tmp/out.mp4'))
    print()
