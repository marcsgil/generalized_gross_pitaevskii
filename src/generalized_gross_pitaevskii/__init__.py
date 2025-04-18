import os
os.environ["PYTHON_JULIACALL_THREADS"] = "auto"
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

from juliacall import Main as jl
import numpy as np
import juliapkg

juliapkg.require_julia("1.11")
juliapkg.add("GeneralizedGrossPitaevskii", "1c32f1f8-fc12-41ee-a188-d96e58f04b51", version="0.1",
             url="https://github.com/marcsgil/GeneralizedGrossPitaevskii.jl")

jl.seval("using GeneralizedGrossPitaevskii")


def python2julia(x):
    return np.transpose(x)

def julia2python(x):
    return np.array(np.transpose(x), copy=False)

additiveIdentity = jl.GeneralizedGrossPitaevskii.additiveIdentity

class GrossPitaevskiiProblem:
    def __init__(self, u0, lengths, dispersion=additiveIdentity, potential=additiveIdentity,
                 nonlinearity=additiveIdentity, pump=additiveIdentity, noise_func=additiveIdentity, noise_prototype=additiveIdentity,
                 param=None):

        new_u0 = tuple(map(python2julia, u0))
        self.prob = jl.GrossPitaevskiiProblem(new_u0, lengths, dispersion=dispersion,
                                              potential=potential, nonlinearity=nonlinearity, pump=pump,
                                              noise_func=noise_func, noise_prototype=noise_prototype, param=param)

    def __str__(self):
        N = len(self.prob.lengths)
        return f"{N}D GrossPitaevskiiProblem"

class StrangSplitting:
    def __init__(self):
        self.alg = jl.StrangSplitting()


def solve(prob, alg, tspan, *, dt, nsaves, save_start=True, show_progress=True):
    ts, sol = jl.solve(prob.prob, alg.alg, tspan, dt=dt, nsaves=nsaves,
                       save_start=save_start, show_progress=show_progress)

    return julia2python(ts), tuple(map(julia2python, sol))
