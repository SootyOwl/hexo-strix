import json

p = 'runs/gine-mini/4l-128p32v-jkcat-rel2/s4-6-8/sprt_state.json'
d = json.load(open(p)); s = d['state']; o = s['outcomes']
KEEP = 3000
drop = len(o) - KEEP; drop -= drop % 2      # drop an even count from the FRONT
o = o[drop:]                                 # keeps (P1,P2) pair phase aligned
sc = {'W': 1.0, 'D': 0.5, 'L': 0.0}
n = len(o) // 2
pst = sum(sc[o[2*i]] + sc[o[2*i+1]] for i in range(n))
s.update(outcomes=o, games=len(o), wins=o.count('W'), draws=o.count('D'),
         losses=o.count('L'), total_score=o.count('W') + 0.5*o.count('D'),
         pairs=n, pair_score_total=pst,
         llr=2*(0.525-0.5)/0.5 * (pst - n*1.025))   # your s0/s1/pair_variance
json.dump(d, open(p, 'w'), indent=1)

