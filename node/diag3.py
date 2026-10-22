import sys, subprocess, os
sys.path.insert(0, '/mnt/workspace/node1/dream/tools')
import dream_loop as dl

m, t = dl.build_model('/mnt/workspace/node1/models/Qwen2.5-0.5B-Instruct', None)
for i in [0, 2, 5]:
    s = dl.DREAM_BANK[i]
    p = dl.build_prompt(s['task'], s.get('hint', ''), [])
    g = dl.gen_code(m, t, p, max_new=512, temp=0.0)
    lang, code = dl.extract_code(g)
    code2 = dl.inject_imports(code)
    print("### task[%d] %s  origlen=%d" % (i, s['task'][:38], len(code or '')))
    print("INJECTED (first 480):\n" + code2[:480])
    fn = '/tmp/d3_%d.py' % i
    open(fn, 'w').write(code2)
    r = subprocess.run([sys.executable, fn], capture_output=True, text=True,
                       timeout=90, cwd='/tmp', env={**os.environ, 'MPLBACKEND': 'Agg'})
    print("rc=", r.returncode, "ERR_TAIL:", r.stderr.strip().splitlines()[-3:])
    print()
