# Object Oriented Programming (OOP) in Python - Day 4 Notes

This document provides a comprehensive explanation of the topics covered in the "G3_day_4.ipynb" notebook, focusing on Object Oriented Programming (OOP) in Python. Each concept is accompanied by detailed explanations and practical examples.

---

## 1. What is Object Oriented Programming (OOP)?

**Object Oriented Programming (OOP)** is a programming paradigm that organizes software design around objects. An object is a collection of data (attributes) and behaviors (methods/functions) that represent real-world entities.

### Key Concepts:
- **Objects**: Instances that contain data (attributes) and functions (methods).
- **Attributes**: Variables that hold data specific to an object.
- **Methods**: Functions that define the behaviors of an object.

#### Example: Object with Attributes and Method
```python
name = "Arun Singh"

def show_name():
    name = "Ajay"
    return name

print(show_name())  # Output: Ajay
```
In this example, the `show_name` function defines a local variable `name` and returns it.

---

## 2. Classes and Objects in Python

### 2.1 What is a Class?

A **class** is like a blueprint for creating objects. It defines a set of attributes and methods that the created objects will have.

#### Example: Defining a Simple Class
```python
class Visitor:
    name = "John"  # Attribute within the class
```

### 2.2 What is an Object?

An **object** is an instance of a class. When you create an object, you are creating an entity that follows the blueprint defined by the class.

#### Example: Creating an Object and Accessing Attributes
```python
v1 = Visitor()      # Creating an object of class Visitor
print(v1.name)      # Output: John
```

---

## 3. Methods in Classes

### 3.1 Defining Methods

**Methods** are functions defined inside a class. The first parameter of a method is always `self`, which refers to the current object.

#### Example: Class with a Method
```python
class Employee:
    def emp_details(self):  # 'self' refers to the current object
        return {"name": "Rajesh", "age": 22}

e1 = Employee()
print(e1.emp_details())  # Output: {'name': 'Rajesh', 'age': 22}
```

#### Common Error Example
Calling a method without an object will result in an error:
```python
emp_details()  # NameError: name 'emp_details' is not defined
```

---

## 4. Why Use OOP?

OOP offers several advantages in software development:

- **Better Organization**: Code is structured around real-world entities.
- **Reusability**: Classes and inheritance enable code reuse.
- **Modularity**: Code is divided into logical sections (classes).
- **Maintainability**: Easier to update and maintain code.
- **Real-World Modeling**: Objects match the way we think about the world.

---

## 5. Practical Example: Visitors Class

Let's build a more advanced example that demonstrates multiple OOP features.

### 5.1 Defining the Visitors Class

```python
class Visitors:
    def visitor_details(self, name, age, city):
        self.details = {"Name": name, "Age": age, "City": city}

    def add_department(self, dept):
        self.details['department'] = dept

    def show_visitor_details(self):
        return self.details
```

### 5.2 Creating and Using Objects

```python
v1 = Visitors()
v2 = Visitors()
v3 = Visitors()

v1.visitor_details("Raghav", 23, "Delhi")
v2.visitor_details("Suraj", 33, "Coimbatore")
v3.visitor_details("Rakhi", 37, "Indore")

print(v1.show_visitor_details())
# Output: {'Name': 'Raghav', 'Age': 23, 'City': 'Delhi'}

print(v2.show_visitor_details())
# Output: {'Name': 'Suraj', 'Age': 33, 'City': 'Coimbatore'}

print(v3.show_visitor_details())
# Output: {'Name': 'Rakhi', 'Age': 37, 'City': 'Indore'}
```

### 5.3 Adding More Information

```python
v1.add_department("Training & Development")
v2.add_department("IT")
v3.add_department("Sales")

print(v1.show_visitor_details())
# Output: {'Name': 'Raghav', 'Age': 23, 'City': 'Delhi', 'department': 'Training & Development'}

print(v2.show_visitor_details())
# Output: {'Name': 'Suraj', 'Age': 33, 'City': 'Coimbatore', 'department': 'IT'}

print(v3.show_visitor_details())
# Output: {'Name': 'Rakhi', 'Age': 37, 'City': 'Indore', 'department': 'Sales'}
```

---

## 6. Summary

- **OOP** is a powerful programming paradigm that models real-world entities as objects.
- **Classes** are blueprints; **objects** are instances of classes.
- **Attributes** store object data; **methods** define object behavior.
- OOP makes code modular, reusable, and maintainable.
- Practical examples illustrate how to define classes, create objects, and use methods and attributes.

---

## Further Reading

- [Python OOP Documentation](https://docs.python.org/3/tutorial/classes.html)
- [Real Python: OOP in Python](https://realpython.com/python3-object-oriented-programming/)
