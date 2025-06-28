# Python Loops, Functions, and Arguments – Study Notes

---

## 1. User Authentication Logic with Loops

### Example: Password Attempts

```python
for i in range(3):
    keyword = input("Enter Keyword")
    if (keyword == 'amo1234'):
        print("Access granted")
        print("Successfull login")
        break
    else:
        print("Try again")
print("all attempt fail")
```

**Explanation:**
- The loop allows the user three attempts to enter the correct password (`amo1234`).
- If the correct keyword is entered, the loop is broken, and "Access granted" and "Successfull login" are printed.
- If all three attempts fail, "all attempt fail" is printed after the loop.

**Variation with `else` block:**

```python
for i in range(3):
    keyword = input("Enter Keyword")
    if (keyword == 'amo1234'):
        print("Access granted")
        print("Successfull login")
        break
    else:
        print("Try again")
else:
    print("All attempt failed")
```

- The `else` block after a `for` loop runs only if the loop completes without a `break`.

---

## 2. For Loops: Printing Even and Odd Numbers

### Print Odd Numbers Between 1 and 50

```python
# Odd numbers
for i in range(1, 51):
    if (i % 2 == 1):
        print(i)
```

### Print Even Numbers Between 1 and 50

```python
# Even numbers
for i in range(1, 51):
    if (i % 2 == 0):
        print(i)
```

**Explanation:**
- `%` is the modulo operator, which gives the remainder. If a number modulo 2 is 0, it is even; if 1, it is odd.

---

## 3. Nested Loops: Multiplication Table

**Task:** Create a program that displays the multiplication table from 1 to 10.

### Example Output

```
1 * 1 = 1
1 * 2 = 2
...
10 * 10 = 100
```

### Code Example

```python
for x in range(1, 11):
    for i in range(1, 11):
        print(f"{x} * {i} = {x * i}")
    print()
```

**Explanation:**
- The outer loop (`x`) iterates numbers 1 to 10.
- The inner loop (`i`) also iterates 1 to 10.
- Each combination prints the multiplication result, and a blank line separates tables for each number.

---

## 4. Operators: Integer Division and Modulo

### Integer Division

```python
a = 24
a = 23
print(a // 2)  # Output: 11
```
- `//` is the integer division operator, which returns the quotient without the remainder.

### Modulo Operator

```python
print(a % 2)
print(a % 3)
print(a % 4)
print(a % 5)
print(a % 6)
print(a % 7)
print(a % 8)
print(a % 9)
print(a % 10)
print(a % 11)
```

---

## 5. Checking for Prime Numbers

### Step-by-Step Function

#### Print Remainders

```python
def check_prime(x):
    n = x // 2
    for i in range(2, n):
        print(f"{x} % {i} = {x % i}")
```

#### Return True/False for Prime

```python
def check_prime(x):
    n = x // 2
    for i in range(2, n+1):
        if x % i == 0:
            return False
    else:
        return True
```

**Examples:**

```python
print(check_prime(4))  # Output: False
print(check_prime(11)) # Output: True
```

---

## 6. Functions in Python

### Function with Default Arguments

```python
def emp_details(name='NA', age='NA', email='NA'):
    emp_info = {
        "Name": name,
        "Age": age,
        "Email": email
    }
    return emp_info
```

**Usage Examples:**

```python
emp_details("Pavan", 25, 'pavan@abc.com')
# Output: {'Name': 'Pavan', 'Age': 25, 'Email': 'pavan@abc.com'}

emp_details("Pavan", 25)
# Output: {'Name': 'Pavan', 'Age': 25, 'Email': 'NA'}

emp_details("Jai")
# Output: {'Name': 'Jai', 'Age': 'NA', 'Email': 'NA'}
```

---

### Positional Arguments

Arguments are assigned based on their position.

```python
emp_details("Pavan", 25, 'pavan@abc.com')
```

### Keyword Arguments

Arguments are specified by the parameter name.

```python
emp_details(email='pavan@abc.com', name="Pavan", age=25)
```

---

## 7. Variable Length Arguments

### `*args` - Variable Length Positional Arguments

```python
def emp_salaries(*salary):
    for val in salary:
        val = val + val*0.1
        print(val)
```

**Examples:**

```python
emp_salaries(65000, 55500)
# Output:
# 71500.0
# 61050.0

emp_salaries(65000, 55500, 35000)
# Output:
# 71500.0
# 61050.0
# 38500.0
```

### `**kwargs` - Variable Length Keyword Arguments

```python
def student_details(**kwargs):
    for key, value in kwargs.items():
        print(f"{key} - {value}")
```

**Example:**

```python
student_details(name="Ajay", course="Python", Duration="4 Weeks")
# Output:
# name - Ajay
# course - Python
# Duration - 4 Weeks
```

---

## Summary

- **Loops**: Use for and while to repeat actions, including nested loops for tables.
- **Operators**: `%` for remainder, `//` for integer division.
- **Functions**: Support default, positional, and keyword arguments.
- **Variable Arguments**: Use `*args` for positional and `**kwargs` for keyword argument lists.

---
# Exception Handling & File Handling in Python

## Exception Handling

Exception handling in Python is used to manage errors that occur during program execution. It helps prevent the abrupt termination of a program and allows you to handle errors gracefully.

### Basic Syntax

```python
try:
    # Code that may raise an exception
    x = 1 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
finally:
    print("This block always executes.")
```

### Common Exception Types

- `ZeroDivisionError`: Raised when division by zero occurs.
- `ValueError`: Raised when a function receives an argument of the correct type but inappropriate value.
- `FileNotFoundError`: Raised when a file or directory is requested but does not exist.
- `TypeError`: Raised when an operation or function is applied to an object of inappropriate type.

### Multiple Except Blocks

You can handle different exceptions with multiple except blocks:

```python
try:
    num = int(input("Enter a number: "))
    result = 10 / num
except ValueError:
    print("Please enter a valid integer.")
except ZeroDivisionError:
    print("Cannot divide by zero!")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
else:
    print(f"Result: {result}")
finally:
    print("Execution complete.")
```

## File Handling

File handling allows you to create, read, write, and close files in Python.

### Opening and Closing Files

Use the `open()` function to open a file. Always close the file after completing operations.

```python
file = open("example.txt", "w")
file.write("Hello, World!")
file.close()
```

### Using `with` Statement

The `with` statement handles opening and closing files automatically.

```python
with open("example.txt", "r") as file:
    content = file.read()
    print(content)
```

### Common File Modes

- `'r'`: Read (default mode). File must exist.
- `'w'`: Write. Creates a new file or overwrites existing file.
- `'a'`: Append. Adds content to the end of the file.
- `'b'`: Binary mode.
- `'t'`: Text mode (default).

You can combine modes, e.g., `'rb'` for reading binary files.

### Handling Exceptions in File Operations

It's good practice to include exception handling in file operations to manage errors like missing files or permission issues.

```python
try:
    with open("data.txt", "r") as file:
        data = file.read()
        print(data)
except FileNotFoundError:
    print("File not found. Please check the file name.")
except IOError:
    print("An I/O error occurred.")
```

---

**Best Practices:**
- Always use the `with` statement for file operations.
- Handle exceptions to prevent program crashes.
- Use specific exception types for clarity.

**Practice these examples and modify them to deepen your understanding!**
