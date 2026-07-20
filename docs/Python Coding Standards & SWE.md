## Python Coding Standards & SWE Principles for Students  
Inspired by Google, Microsoft, and Industry Best Practices

Objective  
To help students write clean, safe, and professional Python code by following simple,  
real-world coding standards and software engineering principles.

1. Formatting & Layout  
• Use 4 spaces per indentation level (never tabs).  
• Limit lines to 80 characters.  
• Use blank lines to separate functions/classes.  
• Align hanging indents with the opening delimiter.  
# Good  
def long_function_name(  
var_one, var_two, var_three,  
var_four):  
pass  
# Bad  
def long_function_name(var_one, var_two,  
var_three, var_four): pass

2. Naming Conventions  
• Variables, functions, methods: snake case  
• Classes: CamelCase  
• Constants: UPPER SNAKE CASE  
• Private members: start with  
 

student_marks = [90, 80, 70]  
class StudentRecord:  
pass  
MAX_SCORE = 100  
def _helper_function():  
pass

3. Comments & Documentation  
• Use # for single-line comments.  
• Use triple quotes for multi-line comments.  
• Use docstrings (Google style) for all public modules, classes, and functions.  
# This is a single-line comment  
"""  
This is a multi-line comment.  
It can span several lines.  
"""  

Google-Style Docstring Example  
def get_grade(score: int) -> str:  
"""Returns the grade for a given score.  
Args:  
score (int): The student’s score (0-100).  
Returns:  
str: The grade (’A’, ’B’, ’C’, ’D’, ’F’).  
Raises:  
ValueError: If score is not between 0 and 100.  
Examples:  
>>> get_grade(95)  
’A’  
>>> get_grade(50)  
’F’  
"""  
if not 0 <= score <= 100:  
raise ValueError("Score must be between 0 and 100")  
if score >= 90:  
return ’A’  
elif score >= 80:  
return ’B’  
elif score >= 70:  
return ’C’  
elif score >= 60:  
return ’D’  
else:  
return ’F’

4. Function & Method Design  
• Use verbs for function names.  
• One function, one job (single responsibility).  
• Prefer short functions (<20 lines).  
• Limit arguments (1-3 preferred).  
• Avoid flag arguments; split into separate functions instead.

def fetch_students():  
pass  
def display_students(students):  
pass  
# Bad (does two things)  
def fetch_and_display_students():  
pass  
# Bad (flag parameter)  
def transform(text, uppercase):  
return text.upper() if uppercase else text.lower()  
# Good (split)  
def to_uppercase(text):  
return text.upper()  
def to_lowercase(text):  
return text.lower()

5. Error Handling  
• Use try-except for error handling.  
• Catch specific exceptions.  
• Provide clear error messages.  
• Use custom exceptions for business logic errors.  
try:  
result = 10 / value  
except ZeroDivisionError:  
print("Cannot divide by zero.")  
except TypeError:  
print("Invalid type provided.")  
raise  
class StudentNotFoundError(Exception):  
pass

6. Testing  
• Write unit tests for every function.  
• Test edge cases: empty lists, large inputs, invalid data.  
Software Engineering for Students Python Coding Standards  
• Use parameterized tests for multiple cases.  
• Aim for 80%+ coverage.  
def average(numbers):  
if not numbers:  
raise ValueError("Cannot compute the average of an empty list.")  
return sum(numbers) / len(numbers)  
# Edge case tests  
assert average([10]) == 10  
try:  
average([])  
except ValueError:  
pass

7. Security Best Practices  
• Validate all inputs (length, type, content).  
• Use parameterized queries for databases.  
• Never hardcode secrets in code.  
def update_email(user_id, new_email):  
if "@" not in new_email:  
raise ValueError("Invalid email address")  
cursor.execute("UPDATE users SET email = %s WHERE id = %s", (new_email,  
user_id))

8. Version Control (Git)  
• Make small, focused commits with clear messages.  
• Use branches for features/bugs.  
• Review code before merging.  
git commit -m "fix: handle empty student list in average()"

9. Principles & Best Practices  
• DRY (Don’t Repeat Yourself): Reuse code via functions, classes, and modules.  
• KISS (Keep It Simple, Stupid): Prefer simple, readable solutions.  
• YAGNI (You Aren’t Gonna Need It): Don’t add features ”just in case”.  
Software Engineering for Students Python Coding Standards  
• SOLID (for OOP): Single Responsibility, Open/Closed, Liskov Substitution, In-  
terface Segregation, Dependency Inversion.  
• Code Readability: Write code as if the next person to read it is a beginner.

10. Checklist for Every Python File  
4-space indentation, 80-char lines  
Descriptive snake case names  
Classes in CamelCase  
Docstrings for all public APIs  
Functions < 20 lines, one purpose  
Specific exception handling  
Unit tests for all logic and edge cases  
No hardcoded secrets, safe DB queries  
Small, clear git commits

11. Summary Table: What to Always Do  
Area Rule/Example  
Indent 4 spaces, never tabs  
Names snake case for vars, CamelCase for classes  
Comments # for short, docstrings for public APIs  
Functions One purpose, <20 lines, verbs in names  
Errors Specific excepts, clear messages  
Tests Edge cases, parameterized, >80% coverage  
Security Validate input, never hardcode secrets  
Git Small commits, clear messages, code reviews

”Code is read much more often than it is written.”  
— Guido van Rossum

## IMPORTANT:

Checklist for Every Python File:  
• Use Git for version control: commit regularly with clear messages.  
• Write unit tests for every function.  
• Test edge cases: empty lists, large inputs, invalid data.  
• Use parameterized tests for multiple cases.  
• Aim for 80%+ coverage  
• Use # for single-line comments.  
• Use triple quotes for multi-line comments.  
• Use docstrings (Google style) for all public modules, classes, and functions.  
• Use verbs for function names.  
• One function, one job (single responsibility).  
• Prefer short functions (<20 lines).  
• Limit arguments (1-3 preferred).  
• Avoid flag arguments; split into separate functions instead.  
• Use try-except for error handling.  
• Catch specific exceptions.  
• Provide clear error messages.  
• Use custom exceptions for business logic errors.  
• Validate all inputs (length, type, content).  
• Use parameterized queries for databases.  
• Organize your code into logical modules/packages.  
• DRY (Don’t Repeat Yourself): Reuse code via functions, classes, and modules.  
• KISS (Keep It Simple, Stupid): Prefer simple, readable solutions.  
• YAGNI (You Aren’t Gonna Need It): Don’t add features ”just in case”.  
• SOLID (for OOP): Single Responsibility, Open/Closed, Liskov Substitution, In-  
terface Segregation, Dependency Inversion.  
• Code Readability: Write code as if the next person to read it is a beginner.

# IMPORTANT:  
Loose Coupling	Classes/modules should know as little as possible about each other  
Single Responsibility	Each class should do one thing well  
DRY + KISS	No duplicate logic. Keep everything simple and modular. Avoid Tight Coupling

