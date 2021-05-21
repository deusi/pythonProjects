Name: Denis Rybkin
Student ID: 5404826
Email: rybki001@umn.edu

Instructions: Simply go to the code directory (in this case it is HW_Code)
and type "python3 q3i.py", "python3 q3ii.py" or "python3 q4.py" in the console,
depending on the question and the code you are trying to test. It is
recommended using "python3 -W ignore your_filename" (for q3i.py, q3ii.py
or q4.py instead of your_filename), since current implementation throws
a bunch of warnings related to a small number of iterations.
Note: For Windows 10 (on my device), it suffices to type "python -W ignore
your_filename", while some linux based machines require specification, i.e.
"python3 -W ignore your_filename".

Assumptions:
  - Code: The code was implemented only for the usage on the datasets
and the methods given in the questions. I tried to make my code as
flexible as possible, but it obviously doesn't provide the kind
of flexibility that the built-in functions have.
  - System: It is assumed that you have numpy, pandas and scikit-learn
libraries installed on your device (and connected to the cp in case of
Windows). Otherwise, the code will not run and produce an error. It is
also assumed that you use Python 3. The code was written using Python 3.8.3,
numpy 1.18.5, pandas 1.0.5 and scikit-learn: 0.23.1.

Unnecessary details: Some of the code wasn't written in the most
space efficient fashion and ideally, some of the functions should
be grouped together and renamed. Since the code produces correct
output and this course is not about writing clean python code,
it was decided to leave it as it is. (also because troubleshooting
would take some time and I have lots of other projects)
