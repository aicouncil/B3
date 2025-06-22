# Concepts Covered in "G3_day_2.ipynb"

This document summarizes the key programming concepts introduced in the file `G3_day_2.ipynb`. Each concept is explained with definitions, examples, and notes for clarity.

---

## 1. Operators in Python

Operators are special symbols used to perform operations on variables and values. Python supports several types of operators:

### 1.1 Arithmetic Operators

These are used to perform mathematical operations.

| Operator | Description       | Example      | Output    |
|----------|-------------------|--------------|-----------|
| `+`      | Addition          | `17 + 5`     | `22`      |
| `-`      | Subtraction       | `17 - 5`     | `12`      |
| `*`      | Multiplication    | `17 * 5`     | `85`      |
| `/`      | Division          | `17 / 5`     | `3.4`     |
| `**`     | Exponentiation    | `17 ** 5`    | `1419857` |
| `//`     | Floor Division    | `17 // 5`    | `3`       |
| `%`      | Modulus           | `17 % 5`     | `2`       |

**Example:**
```python
a = 17
b = 5
print(a + b)  # 22
print(a - b)  # 12
print(a * b)  # 85
print(a / b)  # 3.4
print(a ** b) # 1419857
print(a // b) # 3
print(a % b)  # 2
```

---

### 1.2 Assignment Operators

Used to assign values to variables and modify them.

| Operator | Example    | Explanation        |
|----------|------------|-------------------|
| `=`      | `x = 10`   | Assigns 10 to x   |
| `+=`     | `x += 3`   | x = x + 3         |
| `-=`     | `x -= 4`   | x = x - 4         |
| `*=`     | `x *= 3`   | x = x * 3         |
| `/=`     | `x /= 4`   | x = x / 4         |
| `**=`    | `x **= 2`  | x = x ** 2        |
| `//=`    | `x //= 3`  | x = x // 3        |
| `%=`     | `x %= 4`   | x = x % 4         |

**Example:**
```python
x = 10
x += 3  # 13
x -= 4  # 9
x *= 3  # 27
x /= 4  # 6.75
x **= 2 # 45.5625
x //= 3 # 15.0
x %= 4  # 3.0
print(x)
```

---

### 1.3 Comparison Operators

Used to compare two values.

| Operator | Description          | Example         | Output   |
|----------|----------------------|-----------------|----------|
| `==`     | Equal to             | `age == 21`     | `True`   |
| `!=`     | Not equal to         | `age != 21`     | `False`  |
| `>`      | Greater than         | `age > 18`      | `True`   |
| `<`      | Less than            | `age < 18`      | `False`  |
| `>=`     | Greater than or equal| `age >= 18`     | `True`   |
| `<=`     | Less than or equal   | `age <= 18`     | `False`  |

**Example:**
```python
age = 21
print(age == 21)  # True
print(age > 18)   # True
print(age < 18)   # False
```

---

### 1.4 Logical Operators

Combine multiple conditions.

| Operator | Description            | Example                         | Output   |
|----------|------------------------|---------------------------------|----------|
| `and`    | True if both are true  | `age > 18 and weight > 50`      | `True`   |
| `or`     | True if at least one   | `age > 18 or weight < 50`       | `True`   |
| `not`    | Negates the condition  | `not(True)`                     | `False`  |

**Example:**
```python
age = 21
weight = 56
print(age > 18 and weight > 50)  # True
print(age > 18 or weight < 50)   # True
print(not(age < 18))             # True
```

---

### 1.5 Membership Operators

Test if a value is part of a sequence (string, list, etc.).

| Operator | Description             | Example                 | Output   |
|----------|-------------------------|-------------------------|----------|
| `in`     | Value exists in object  | `'nice' in text`        | `True`   |
| `not in` | Value does not exist    | `'good' not in text`    | `True`   |

**Example:**
```python
text = "It was a nice morning"
print('nice' in text)  # True
print('good' in text)  # False

cities = ["Mumbai", "Delhi", "Kolkata"]
print("Delhi" in cities)    # True
print("Jaipur" in cities)   # False
```

---

## 2. Conditional Programming

Conditional programming allows you to execute code based on certain conditions.

### 2.1 Simple If Statement

```python
if True:
    print("Hello World!")  # Output: Hello World!
```

### 2.2 If-Else Statement

```python
age = 13
if age > 18:
    print("You are eligible to vote")
else:
    print("You are not eligible to vote as of now")
```

### 2.3 If-Elif-Else Statement

Used for multiple conditional branches.

```python
marks = int(input("Enter your marks"))
if marks > 90:
    print("Grade A+")
elif marks > 80:
    print("Grade A")
elif marks > 70:
    print("Grade B+")
elif marks > 60:
    print("Grade B")
elif marks > 50:
    print("Grade C+")
elif marks > 40:
    print("Grade C")
else:
    print("Failed")
```

### 2.4 Nested If Statements

```python
age = int(input("Enter your age"))
if age > 18:
    if age <= 30:
        print("You are eligible for admission")
    else:
        print("You are not eligible for admission")
else:
    print("Not eligible for admission")
```

---

## 3. Practical Examples

### 3.1 Voting Eligibility

```python
age = int(input("Enter your age: "))
if age > 18:
    print("You are eligible to vote")
else:
    print("You are not eligible to vote")
```

### 3.2 User Login Authentication

```python
userid = input("Enter the user Id")
userpassword = input("Enter the password")

if (userid == 'admin' and userpassword == 'password'):
    print("Login Successful")
else:
    print("Login Failed")
    print("try again!!")
```

---

## Notes

- User input in Python via `input()` is always a string. Use `int()` to convert to integer if needed.
- Indentation is crucial in Python to define code blocks.
- Operators and conditionals are foundational for all Python programming.

---

**End of Document**
