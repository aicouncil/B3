# G3 Day 1: Python Basics – Concepts, Explanations & Examples

This document covers all concepts introduced in the `G3_day_1.ipynb` notebook, with explanations and illustrated examples for each topic.

---

## 1. Variables and Values

A variable is a name that refers to a value stored in memory. You can assign values to variables using the `=` operator.

**Example:**
```python
name = "Bipul"
age = 35
city = 'Delhi'
```
You can print multiple variables together:
```python
print(name, age, city)
# Output: Bipul 35 Delhi

print("My name is", name, "and age is", age, "and city is", city)
# Output: My name is Bipul and age is 35 and city is Delhi
```

**Multiple Assignments:**
```python
x, y, z = 21, 34, 2
print(x)  # 21
print(y)  # 34
print(z)  # 2

a = b = c = 7
print(a)  # 7
print(b)  # 7
print(c)  # 7
```

---

## 2. Data Types & Data Structures

Python has several built-in data types, such as:

- **str** (String)
- **int** (Integer)
- **float** (Floating-point number)

**Example:**
```python
emp_name = "Ajit"
emp_age = 23
emp_salary = 23000.76

print(emp_name, type(emp_name))    # Ajit <class 'str'>
print(emp_age, type(emp_age))      # 23 <class 'int'>
print(emp_salary, type(emp_salary))# 23000.76 <class 'float'>
```

You can perform operations with these variables:
```python
a = 7
print(a * 2)  # Output: 14
```

---

## 3. Strings

Strings are sequences of characters and are **immutable** (cannot be changed after creation).

**Examples:**

- Creating and printing a string:
    ```python
    review = "It was nice movie"
    print(review)  # It was nice movie
    ```

- Accessing characters by index:
    ```python
    print(review[0])  # I
    print(review[1])  # t
    ```

- **Immutability:** You cannot change a character by assignment.
    ```python
    review[0] = 'K'  # Raises TypeError
    ```

- String length:
    ```python
    print(len(review))  # Output: 17
    ```

### String Operations

- Convert to upper/lowercase:
    ```python
    print(review.upper())  # IT WAS NICE MOVIE
    print(review.lower())  # it was nice movie
    ```

- Remove spaces:
    ```python
    name = "   Pavan    "
    print(name.strip())   # Pavan
    print(name.lstrip())  # Pavan    (left spaces removed)
    print(name.rstrip())  #    Pavan (right spaces removed)
    ```

- Slicing:
    ```python
    print(review[0:6])    # It was
    print(review[7:11])   # nice
    ```

- Splitting strings:
    ```python
    print(review.split(" "))          # ['It', 'was', 'nice', 'movie']
    text = "Hello, How are you?"
    print(text.split(","))            # ['Hello', ' How are you?']
    ```

- Joining strings:
    ```python
    text_list = ["Nice", "Movie"]
    print(''.join(text_list))         # NiceMovie

    name = "Ananya Thakur"
    print(name.split(" "))            # ['Ananya', 'Thakur']
    print(''.join(name.split(" ")))   # AnanyaThakur
    ```

- Replacing substrings:
    ```python
    review = "It was nice movie"
    print(review.replace('nice', 'bad'))  # It was bad movie
    print(review.replace('i', 'k'))       # It was nkce movke
    ```

---

## 4. Lists

Lists are **sequential**, **mutable** data structures that can contain elements of any type.

**Examples:**
```python
cities = ["Delhi", "Jaipur", "Mysore", "Banglore", "Pune"]
print(cities)               # ['Delhi', 'Jaipur', 'Mysore', 'Banglore', 'Pune']

print(cities[0])            # Delhi
cities[1] = "Indore"        # Modify element
print(cities)               # ['Delhi', 'Indore', 'Mysore', 'Banglore', 'Pune']
print(len(cities))          # 5

# Slicing
print(cities[0:3])          # ['Delhi', 'Indore', 'Mysore']
print(cities[1:4])          # ['Indore', 'Mysore', 'Banglore']
```

---

## 5. List Operations

- **append(value):** Add an element to the end.
    ```python
    cities.append("Hyderabad")
    print(cities)
    # ['Delhi', 'Indore', 'Mysore', 'Banglore', 'Pune', 'Hyderabad']
    ```

- **remove(value):** Remove the first occurrence of value.
    ```python
    cities.remove("Mysore")
    print(cities)
    # ['Delhi', 'Indore', 'Banglore', 'Pune', 'Hyderabad']
    ```

- **insert(index, value):** Insert at a specific index.
    ```python
    cities.insert(2, "Lucknow")
    print(cities)
    # ['Delhi', 'Indore', 'Lucknow', 'Banglore', 'Pune', 'Hyderabad']
    ```

- **pop(index):** Remove and return element at index.
    ```python
    cities.pop(3)
    print(cities)
    # ['Delhi', 'Indore', 'Lucknow', 'Pune', 'Hyderabad']
    ```

- **index(value):** Get index of first occurrence.
    ```python
    print(cities.index("Pune"))  # Output: 3
    ```

---

## Summary

This notebook introduced the following Python basics:
- Declaring and working with variables
- Understanding core data types (`str`, `int`, `float`)
- String operations and immutability
- List creation, access, mutation, and common operations

Each section included code examples to illustrate the concept in practice. Experiment with these snippets in a Python environment to deepen your understanding!
