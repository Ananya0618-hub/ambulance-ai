"""
Fuzzy triage: Mamdani fuzzy inference system written from scratch (no library),
so every step can be explained in the viva.

Inputs
  severity  0-10   (symptom severity score from the call: chest pain, breathing,
                    bleeding, consciousness ...)
  age       0-100  years
  wait      0-30   minutes the call has already been waiting in the queue
Output
  priority  0-100

Pipeline: fuzzification -> rule evaluation (AND=min, OR=max)
          -> implication (min / clipping) -> aggregation (max)
          -> defuzzification (centroid)
"""
import numpy as np


def trimf(x, a, b, c):
    x = np.asarray(x, dtype=float)
    left = np.where(b > a, (x - a) / (b - a + 1e-12), (x >= b).astype(float))
    right = np.where(c > b, (c - x) / (c - b + 1e-12), (x <= b).astype(float))
    return np.clip(np.minimum(left, right), 0, 1)


def trapmf(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    left = np.ones_like(x) if b == a else (x - a) / (b - a)
    right = np.ones_like(x) if d == c else (d - x) / (d - c)
    y = np.minimum(np.minimum(left, 1.0), right)
    if b == a:
        y = np.where(x < a, 0.0, y)
    if d == c:
        y = np.where(x > d, 0.0, y)
    return np.clip(y, 0, 1)


class FuzzyTriage:
    def __init__(self):
        self.universe = {
            "severity": np.linspace(0, 10, 201),
            "age": np.linspace(0, 100, 201),
            "wait": np.linspace(0, 30, 201),
            "priority": np.linspace(0, 100, 501),
        }
        # Membership functions: name -> (type, params)
        self.mfs = {
            "severity": {
                "low": ("trap", (0, 0, 2, 4)),
                "medium": ("tri", (3, 5, 7)),
                "high": ("trap", (6, 8, 10, 10)),
            },
            "age": {
                "child": ("trap", (0, 0, 8, 15)),
                "adult": ("trap", (12, 20, 50, 62)),
                "elderly": ("trap", (55, 68, 100, 100)),
            },
            "wait": {
                "short": ("trap", (0, 0, 3, 7)),
                "medium": ("tri", (5, 10, 15)),
                "long": ("trap", (12, 18, 30, 30)),
            },
            "priority": {
                "low": ("trap", (0, 0, 15, 35)),
                "medium": ("tri", (25, 45, 65)),
                "high": ("tri", (55, 72, 88)),
                "critical": ("trap", (80, 92, 100, 100)),
            },
        }
        # Rule base: (description, function(memberships) -> firing strength, output term)
        self.rules = [
            ("IF severity is high THEN critical",
             lambda m: m["severity"]["high"], "critical"),
            ("IF severity is medium AND age is (child OR elderly) THEN high",
             lambda m: min(m["severity"]["medium"],
                           max(m["age"]["child"], m["age"]["elderly"])), "high"),
            ("IF severity is medium AND age is adult THEN medium",
             lambda m: min(m["severity"]["medium"], m["age"]["adult"]), "medium"),
            ("IF severity is medium AND wait is long THEN high",
             lambda m: min(m["severity"]["medium"], m["wait"]["long"]), "high"),
            ("IF severity is low AND age is adult AND wait is (short OR medium) THEN low",
             lambda m: min(m["severity"]["low"], m["age"]["adult"],
                           max(m["wait"]["short"], m["wait"]["medium"])), "low"),
            ("IF severity is low AND age is (child OR elderly) THEN medium",
             lambda m: min(m["severity"]["low"],
                           max(m["age"]["child"], m["age"]["elderly"])), "medium"),
            ("IF severity is low AND wait is long THEN medium",
             lambda m: min(m["severity"]["low"], m["wait"]["long"]), "medium"),
        ]

    def mf(self, var, term, x):
        kind, p = self.mfs[var][term]
        return trimf(x, *p) if kind == "tri" else trapmf(x, *p)

    def fuzzify(self, severity, age, wait):
        crisp = {"severity": np.clip(severity, 0, 10),
                 "age": np.clip(age, 0, 100),
                 "wait": np.clip(wait, 0, 30)}
        return {var: {t: float(self.mf(var, t, crisp[var])) for t in self.mfs[var]}
                for var in crisp}

    def infer(self, severity, age, wait, explain=False):
        m = self.fuzzify(severity, age, wait)
        y = self.universe["priority"]
        aggregated = np.zeros_like(y)
        fired = []
        for desc, fn, out in self.rules:
            strength = fn(m)
            if strength > 0:
                fired.append((desc, round(strength, 3)))
                clipped = np.minimum(strength, self.mf("priority", out, y))  # implication
                aggregated = np.maximum(aggregated, clipped)                 # aggregation
        if aggregated.sum() == 0:
            score = 0.0
        else:
            score = float((y * aggregated).sum() / aggregated.sum())        # centroid
        if explain:
            return score, {"memberships": m, "fired_rules": fired,
                           "aggregated": aggregated}
        return score

    def priority(self, severity, age, wait):
        return self.infer(severity, age, wait)


if __name__ == "__main__":
    ft = FuzzyTriage()
    cases = [("Cardiac arrest, 70y", 9.5, 70, 0),
             ("Fracture, 8y child", 5, 8, 2),
             ("Fracture, 30y adult", 5, 30, 2),
             ("Minor cut, 30y, waited 20 min", 2, 30, 20),
             ("Minor cut, 30y, just called", 2, 30, 0),
             ("Fever, 80y", 3, 80, 5)]
    for name, s, a, w in cases:
        score, info = ft.infer(s, a, w, explain=True)
        print(f"{name:32s} -> priority {score:5.1f}   rules: {[r[0].split('THEN')[1].strip() for r in info['fired_rules']]}")
