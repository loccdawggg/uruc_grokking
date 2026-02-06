# URUC Grokking

Continuous grokking in constrained 3-variable dynamics (URUC framework).

## Features
- Sharp grokking onset (~1000 steps)
- Low manifold error (~1e-3 to 1e-8)
- Persistent CIC invariant (~0.115)
- Grok-tuned stability: H cap, adapt scaling, p=3, RMSprop + momentum, clip norm, weight decay

## Run
```bash
pip install numpy matplotlib
python uruc_grokking.py