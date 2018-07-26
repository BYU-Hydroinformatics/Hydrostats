import sympy as sp

latex_symbols = {
    'me': r'$$ME = \frac{1}{n} \sum_{i=0}^{n} (Sim_i - Obs_i)$$',
}

sp.preview(latex_symbols['me'], viewer='file', filename='test.png')
