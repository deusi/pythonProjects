Name: Denis Rybkin
Student ID: 5404826
Email: rybki001@umn.edu

Instructions:
- For Problem 3: Simply go to the code directory (in this case it is HW4_Code)
and type "python3 q3.py" in the console. It is recommended using
"python3 -W ignore q3.py", since current implementation throws
a bunch of warnings related to a small number of iterations.
Note: For Windows 10 (on my device), it suffices to type "python -W ignore
q3.py", while some linux based machines require specification, i.e.
"python3 -W ignore q3.py".

Python Version: 3.8.3
Numpy Version: 1.18.5
Scikit-learn Version: 0.23.1

Required files (provided in the HW3_Code folder):
- q3.py
- mySVM2.py
- my_cross_val.py
- datasets.py

Assumptions:
- Code: The code was implemented only for the usage on the datasets
and the methods given in the question. I tried to make my code as
flexible as possible, but it obviously doesn't provide the kind
of flexibility that the built-in functions have. In general, you shouldn't
try to break the code (with unreasonable values and so on), as it
most likely WILL break.
- System: It is assumed that you have numpy and scikit-learn
libraries installed on your device (and connected to the cp in case of
Windows). Otherwise, the code will not run and produce an error.

Unnecessary details: Some of the code wasn't written in the most
space efficient fashion and some of the functions should
be grouped together and renamed. Since the code produces correct
output and this course is not about writing clean python code,
it was decided to leave it as it is. (also because troubleshooting
would take some time and I have lots of other projects).
The code might not be optimal, but it produces decent results in
reasonable time. I decided to implement mySVM2 based on
myLogisticRegression2 code shared with us, since it produced slightly
better results than my own version.
