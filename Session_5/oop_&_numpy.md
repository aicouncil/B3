# Python OOP and Numpy: A Comprehensive Tutorial

This tutorial covers key concepts in Python Object-Oriented Programming (OOP) and introduces fundamental Numpy operations for data science. Each topic is explained with clear examples and detailed breakdowns.

---

## Section 1: Introduction to OOP in Python

### What is OOP?

Object-Oriented Programming (OOP) is a programming paradigm centered around "objects", which are instances of classes. OOP allows for modeling real-world entities and organizing code into reusable components.

### Constructors and Initializers

A constructor (or initializer) in Python is a special method called `__init__`, which runs as soon as an object of a class is instantiated. It is used to initialize object attributes.

**Example: Defining a Class with a Constructor**

```python
class Visitors:
    def __init__(self, name, age, city):  # constructor
        self.details = {"Name": name, "Age": age, "City": city}

    def add_department(self, dept):
        self.details['department'] = dept

    def show_visitor_details(self):
        return self.details
```

**Creating and Using Objects**

```python
v1 = Visitors("Ram", 22, "Delhi")
v2 = Visitors("Jai", 19, "Jaipur")
v3 = Visitors("Gagan", 21, "Indore")

print(v1.show_visitor_details())  # {'Name': 'Ram', 'Age': 22, 'City': 'Delhi'}
```

---

## Section 2: Practical Class Exercise – BankAccount

### Task: Model a Simple Bank Account

Let's create a `BankAccount` class that supports deposits, withdrawals, and balance display.

**Class Structure**

- Attributes: `account_number`, `account_holder`, `balance`
- Methods:
  - `deposit(amount)`: Adds amount to balance
  - `withdraw(amount)`: Deducts amount if sufficient funds, else notifies
  - `display_balance()`: Shows current balance and account holder

**Implementation Example**

```python
class BankAccount:
    def __init__(self, account_number, account_holder, balance):
        self.account_number = account_number
        self.account_holder = account_holder
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        if amount < self.balance:
            self.balance -= amount
            return "Withdrawal successful"
        else:
            return "Insufficient balance"

    def display_balance(self):
        return {"Name": self.account_holder, "Balance": self.balance}
```

**Demonstration**

```python
rahul = BankAccount("9754357785", "Rahul", 20000)
anita = BankAccount("9754557854", "Anita", 15000)

rahul.deposit(40000)
print(rahul.display_balance())  # {'Name': 'Rahul', 'Balance': 60000}

status = anita.withdraw(5000)
print(status)                   # Withdrawal successful
print(anita.display_balance())  # {'Name': 'Anita', 'Balance': 10000}

status = anita.withdraw(50000)
print(status)                   # Insufficient balance
```

---

## Section 3: Python for Data Science – Numpy Introduction

### Why Use Numpy?

Numpy (Numerical Python) is a library for efficient numerical computations in Python. Unlike basic Python lists, Numpy arrays enable fast, vectorized operations and advanced mathematical functions.

**Scalar and List Multiplication in Python**

```python
val = 20
print(val * 3)  # 60

values = [5, 8, 3, 2, 7]
print(values * 2)  # [5, 8, 3, 2, 7, 5, 8, 3, 2, 7]
```
Note: Multiplying a list repeats the list; it does not multiply each element.

**Multiplying List Elements (Pure Python)**

```python
def perform_multiplication(values, num):
    output = []
    for val in values:
        output.append(val * num)
    return output

print(perform_multiplication([7, 9, 4, 3], 2))  # [14, 18, 8, 6]
```

**Transition to Numpy Arrays**

Numpy arrays allow element-wise operations:

```python
import numpy as np

n1 = np.array([7, 9, 4, 3, 8, 3, 6])
print(n1 * 2)  # [14 18  8  6 16  6 12]
```

---

## Section 4: Numpy Array Operations and Properties

### Creating and Inspecting Numpy Arrays

```python
n2 = np.array([[5, 8, 3, 2, 7], [2, 4, 5, 1, 6], [3, 4, 6, 7, 8]])
print(n2)
# Output:
# [[5 8 3 2 7]
#  [2 4 5 1 6]
#  [3 4 6 7 8]]
```

**Array Properties**

- `ndim`: Number of array dimensions
- `shape`: Shape of the array (rows, columns)
- `size`: Total number of elements

```python
print(n2.ndim)   # 2
print(n2.shape)  # (3, 5)
print(n2.size)   # 15
```

### Array Operations

```python
print(np.mean(n2))     # Average of all elements
print(np.sum(n2))      # Sum of all elements
print(np.prod(n2))     # Product of all elements
print(np.min(n2))      # Minimum value
print(np.max(n2))      # Maximum value
print(np.argmin(n2))   # Index of minimum value (flattened)
print(np.argmax(n2))   # Index of maximum value (flattened)
```

### Axis-Based Operations

- `axis=0` computes along columns
- `axis=1` computes along rows

**Row-wise Operations**

```python
print(np.mean(n2, axis=1))  # Mean for each row
print(np.sum(n2, axis=1))   # Sum for each row
```

**Column-wise Operations**

```python
print(np.mean(n2, axis=0))  # Mean for each column
print(np.sum(n2, axis=0))   # Sum for each column
```

### Indexing and Slicing

Access elements by row and column indices:

```python
print(n2[0, 0])  # First row, first column (5)
print(n2[1, 2])  # Second row, third column (5)
```

---

## Section 5: Numpy Array Creation, Reshaping, and Special Arrays

### Creating Special Arrays

```python
print(np.zeros((4, 3)))
# 4x3 array filled with zeros

print(np.ones((4, 3)))
# 4x3 array filled with ones

print(np.eye(4))
# 4x4 identity matrix
```

### Creating Ranges and Reshaping

```python
n4 = np.arange(0, 36)      # Array with values 0 to 35
print(n4)
# [0, 1, 2, ..., 35]

n5 = n4.reshape(6, 6)      # Reshape to 6x6 array
print(n5)
```

**Valid reshape dimensions for 36 elements:**
- (1, 36), (36, 1), (2, 18), (18, 2), (3, 12), (12, 3), (4, 9), (9, 4), (6, 6)

---

## Summary of Concepts

- **Object-Oriented Programming:** Classes, constructors, methods, real-world modeling
- **Numpy Basics:** Array creation, properties, vectorized operations
- **Array Operations:** Statistical and mathematical methods, axis-based calculations
- **Special Arrays:** Zeros, ones, identity, reshaping
- **Practical Examples:** Bank account simulation, array manipulations

---

**Further Reading:**
- [Python OOP Documentation](https://docs.python.org/3/tutorial/classes.html)
- [Numpy User Guide](https://numpy.org/doc/stable/user/)
