"""Simulation study for the Evidence-Grounded Learner model (EGL).

Synthetic learners have a known latent state over three skills:
  Ar = arrays and indexing (prerequisite)
  Bc = binary search, conceptual facet (why halving works, which half to discard)
  Bi = binary search, implementation facet (bounds, midpoint update, termination)
Items follow a DINA response model (Junker & Sijtsma 2001; de la Torre 2009).
All results are from simulated learners only. No human data.
"""
import itertools, json, sys
import numpy as np

STATES = np.array(list(itertools.product([0, 1], repeat=3)))  # columns Ar, Bc, Bi

def cls(s):
    ar, bc, bi = s
    if ar == 0: return 0          # prerequisite gap (Student C)
    if bc == 1 and bi == 0: return 1  # implementation gap (Student A)
    if bc == 0 and bi == 1: return 2  # conceptual gap (Student B)
    if bc == 0 and bi == 0: return 3  # both facets missing
    return 4                       # mastered
CLASS_NAMES = ["prerequisite", "implementation", "conceptual", "both", "mastered"]
STATE_CLASS = np.array([cls(s) for s in STATES])
NC = 5

# Item bank: name, Q-vector (Ar,Bc,Bi), slip, guess
ITEMS = [
    ("index arithmetic",            (1, 0, 0), 0.10, 0.20),
    ("which half to discard (MCQ)", (1, 1, 0), 0.10, 0.25),
    ("explain the invariant",       (0, 1, 0), 0.15, 0.20),
    ("trace midpoints by hand",     (1, 1, 0), 0.10, 0.10),
    ("fix off-by-one bug",          (1, 0, 1), 0.15, 0.15),
    ("fix termination condition",   (1, 0, 1), 0.15, 0.10),
    ("write binary search (tests)", (1, 1, 1), 0.20, 0.05),
]
Q = np.array([it[1] for it in ITEMS])
SLIP = np.array([it[2] for it in ITEMS]); GUESS = np.array([it[3] for it in ITEMS])
NI = len(ITEMS)
# eta[j, s] = 1 if state s has every skill item j requires
ETA = np.array([[int(np.all(STATES[s] >= Q[j])) for s in range(8)] for j in range(NI)])

def p_correct(j, slip=SLIP, guess=GUESS):
    """P(correct | state) for item j, vector over the 8 states."""
    return np.where(ETA[j] == 1, 1 - slip[j], guess[j])

def prior():
    """Prerequisite-aware prior: facets unlikely without the prerequisite."""
    p = np.zeros(8)
    for i, (ar, bc, bi) in enumerate(STATES):
        pa = 0.7 if ar else 0.3
        pc = (0.5 if ar else 0.1); pi_ = (0.5 if ar else 0.1)
        p[i] = pa * (pc if bc else 1 - pc) * (pi_ if bi else 1 - pi_)
    return p / p.sum()

def update(post, j, y, rel=1.0, w=1.0, slip=SLIP, guess=GUESS):
    """Bayes update. rel = source reliability (mixture with a coin flip);
    w = tempering weight for decayed evidence (likelihood ** w)."""
    pc = p_correct(j, slip, guess)
    lik = pc if y == 1 else 1 - pc
    lik = rel * lik + (1 - rel) * 0.5
    post = post * lik ** w
    return post / post.sum()

def class_post(post):
    return np.bincount(STATE_CLASS, weights=post, minlength=NC)

def H(p):
    p = p[p > 0]; return -(p * np.log2(p)).sum()

def eig(post, j):
    """Expected reduction in entropy of the root-cause class from item j."""
    pc = p_correct(j)
    py1 = (post * pc).sum()
    h0 = H(class_post(post))
    p1 = post * pc; p1 /= p1.sum()
    p0 = post * (1 - pc); p0 /= p0.sum()
    return h0 - (py1 * H(class_post(p1)) + (1 - py1) * H(class_post(p0)))

def respond(rng, state_idx, j, slip=SLIP, guess=GUESS):
    return int(rng.random() < p_correct(j, slip, guess)[state_idx])

TRUE_STATES = {0: (0, 0, 0), 1: (1, 1, 0), 2: (1, 0, 1), 3: (1, 0, 0), 4: (1, 1, 1)}
def state_index(t): return int(np.where((STATES == np.array(t)).all(1))[0][0])

def choose(policy, post, rng, t):
    if policy == "eig": return int(np.argmax([eig(post, j) for j in range(NI)]))
    if policy == "random": return int(rng.integers(NI))
    if policy == "whole_task": return NI - 1
    if policy == "fixed_cycle": return [6, 3, 4, 0, 2, 5, 1][t % 7]
    raise ValueError(policy)

def brier(cp, c):
    y = np.zeros(NC); y[c] = 1; return ((cp - y) ** 2).sum()

# ---------- Experiment 1: probe selection ----------
def exp1(n=3000, T=12, seed=1, true_slip_scale=1.0):
    rng = np.random.default_rng(seed)
    tslip = np.clip(SLIP * true_slip_scale, 0, 0.45); tguess = np.clip(GUESS * true_slip_scale, 0, 0.45)
    out = {}
    for pol in ["eig", "random", "fixed_cycle", "whole_task"]:
        acc = np.zeros(T + 1); br = np.zeros(T + 1); conf_true = []
        for k in range(n):
            c = k % NC; s = state_index(TRUE_STATES[c]); post = prior()
            for t in range(T + 1):
                cp = class_post(post)
                acc[t] += (np.argmax(cp) == c); br[t] += brier(cp, c)
                if t == T:
                    conf_true.append((cp.max(), int(np.argmax(cp) == c)))
                    break
                j = choose(pol, post, rng, t)
                post = update(post, j, respond(rng, s, j, tslip, tguess))
        out[pol] = {"acc": (acc / n).round(4).tolist(), "brier": (br / n).round(4).tolist(),
                    "ece": ece(conf_true)}
    return out

def ece(pairs, bins=10):
    pairs = np.array(pairs); e = 0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (pairs[:, 0] > lo) & (pairs[:, 0] <= hi)
        if m.sum(): e += m.mean() * abs(pairs[m, 0].mean() - pairs[m, 1].mean())
    return round(float(e), 4)

# ---------- Experiment 2: noisy LLM-labelled evidence ----------
def exp2(n=3000, seed=2, eps=0.25, graded=3, noisy=9):
    """graded EIG probes + noisy dialogue-derived labels (flip prob eps)."""
    rng = np.random.default_rng(seed); out = {}
    for mode in ["graded_only", "noisy_as_clean", "noisy_reliability_weighted"]:
        acc = 0; br = 0; pairs = []
        for k in range(n):
            c = k % NC; s = state_index(TRUE_STATES[c]); post = prior()
            for t in range(graded):
                j = choose("eig", post, rng, t); post = update(post, j, respond(rng, s, j))
            if mode != "graded_only":
                for t in range(noisy):
                    j = int(rng.integers(NI)); y = respond(rng, s, j)
                    if rng.random() < eps: y = 1 - y
                    rel = 1.0 if mode == "noisy_as_clean" else 1 - 2 * eps
                    post = update(post, j, y, rel=rel)
            cp = class_post(post); acc += np.argmax(cp) == c; br += brier(cp, c)
            pairs.append((cp.max(), int(np.argmax(cp) == c)))
        out[mode] = {"acc": round(acc / n, 4), "brier": round(br / n, 4), "ece": ece(pairs),
                     "mean_conf": round(float(np.mean([p[0] for p in pairs])), 4)}
    return out

# ---------- Experiment 3: stale memory across sessions ----------
def exp3(n=3000, seed=3, T1=8, T2max=15, w_old=0.3):
    """Session 1 on a prerequisite-gap learner. Between sessions the learner
    either learns arrays (state -> 'both') or does not change. Session 2 counts
    probes until MAP class is correct with confidence >= 0.8."""
    rng = np.random.default_rng(seed); out = {}
    for change in [True, False]:
        res = {}
        for mem in ["carry_full", "decay", "reset"]:
            needed = []
            for k in range(n):
                s1 = state_index(TRUE_STATES[0]); post = prior(); log = []
                for t in range(T1):
                    j = choose("eig", post, rng, t); y = respond(rng, s1, j); post = update(post, j, y); log.append((j, y))
                c2 = 3 if change else 0; s2 = state_index(TRUE_STATES[c2])
                if mem == "carry_full": post2 = post
                elif mem == "reset": post2 = prior()
                else:
                    post2 = prior()
                    for j, y in log: post2 = update(post2, j, y, w=w_old)
                got = T2max + 1
                for t in range(T2max + 1):
                    cp = class_post(post2)
                    if np.argmax(cp) == c2 and cp.max() >= 0.8: got = t; break
                    if t == T2max: break
                    j = choose("eig", post2, rng, t); post2 = update(post2, j, respond(rng, s2, j))
                needed.append(got)
            needed = np.array(needed)
            res[mem] = {"mean_probes": round(float(needed.mean()), 3),
                        "pct_not_resolved_in_15": round(float((needed > T2max).mean() * 100), 2)}
        out["learner_changed" if change else "learner_unchanged"] = res
    return out

# ---------- Experiment 4: contesting a wrong belief ----------
def exp4(n=3000, seed=4, T=6):
    """The tutor holds a confident, wrong-or-right belief that the prerequisite is
    missing (P(Ar=0) ~ 0.9). The learner says 'I already know indexing'.
    honest: truth = implementation gap (Ar=1). mistaken: truth = prerequisite gap."""
    rng = np.random.default_rng(seed); out = {}
    base = prior().copy(); base[STATES[:, 0] == 0] *= 9 / base[STATES[:, 0] == 0].sum()
    base[STATES[:, 0] == 1] *= 1 / base[STATES[:, 0] == 1].sum(); base /= base.sum()
    for who, c in [("honest_claim", 1), ("mistaken_claim", 0)]:
        s = state_index(TRUE_STATES[c]); res = {}
        for pol in ["ignore", "overwrite", "verify"]:
            pr0 = 0; acc3 = 0; accT = 0
            truth_ar = TRUE_STATES[c][0]
            for k in range(n):
                post = base.copy()
                if pol == "overwrite":
                    post[STATES[:, 0] == 0] *= 0.01; post /= post.sum()
                elif pol == "verify":
                    post[STATES[:, 0] == 1] *= 2.0; post /= post.sum()   # weak evidence, LR = 2
                pr0 += post[STATES[:, 0] == truth_ar].sum()
                for t in range(T):
                    j = 0 if (pol == "verify" and t < 2) else choose("eig", post, rng, t)
                    post = update(post, j, respond(rng, s, j))
                    if t == 2: acc3 += np.argmax(class_post(post)) == c
                accT += np.argmax(class_post(post)) == c
            res[pol] = {"p_true_prereq_state_before": round(pr0 / n, 4),
                        "acc_after_3_probes": round(acc3 / n, 4), f"acc_after_{T}_probes": round(accT / n, 4)}
        out[who] = res
    return out

if __name__ == "__main__":
    results = {
        "exp1_well_specified": exp1(),
        "exp1_misspecified_x1.8": exp1(seed=11, true_slip_scale=1.8),
        "exp2_noisy_evidence": exp2(),
        "exp3_stale_memory": exp3(),
        "exp4_contestation": exp4(),
    }
    json.dump(results, open(sys.argv[1] if len(sys.argv) > 1 else "results.json", "w"), indent=1)
    print(json.dumps(results, indent=1))
