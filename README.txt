BKW source package v33

Contents (five files):
  main.tex
  llncs.cls
  preprint.pdf
  README.txt
  verify_classical_grover.py
  verify_walk.py

Compile with a standard LaTeX installation (TeX Live or MiKTeX):
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
All figures and bibliography are embedded in the TeX. Python is not
needed to compile. The standard packages named in the preamble must be
installed; llncs.cls is supplied. No external images or data are needed.

Numerical verification (Python 3):
  python3 -m pip install numpy scipy mpmath
  python3 verify_classical_grover.py
  python3 verify_walk.py

Expected coefficients in 2^(a*n+o(n)):
                    Fresh          Amplified
  Classical         0.888251733    1.635883551
  Grover            0.873673376    1.612982610
  Reusable walk     0.864269772    1.597871236

Optional finer evaluations:
  python3 verify_classical_grover.py --panels 64
  python3 verify_walk.py --degree 48 --quadrature 110 --grade 7 --dps 56

The scripts read no data files, write no files and print their results.
They numerically check the defining equations and reported constants;
they do not prove the heuristic assumptions on intermediate lists.
