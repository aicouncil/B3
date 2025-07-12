# Practical Assignment: OOP, NumPy & Pandas

**Submission Format:** `.ipynb` or `.py` file

---

## SECTION A: Python Object-Oriented Programming

### Q1. Class: BankAccount

Create a class `BankAccount` with the following:

- `__init__` method to initialize `account_holder` and `balance` (default = 0)
- `deposit(self, amount)`
- `withdraw(self, amount)` — only if balance is sufficient
- `display_balance(self)` — prints account holder and balance

**Test using:**
```python
acc1 = BankAccount("Riya", 1500)
acc1.deposit(500)
acc1.withdraw(300)
acc1.display_balance()
```

---

### Q2. Class: Visitor

Create a class `Visitor` with:

- Attributes: `name`, `age`, `purpose`
- Method: `describe()` to print visitor details in a sentence

**Create two visitors and call `describe()` for each.**

---

### Q3. Class with List of Objects: Library

- Create a class `Book` with `title`, `author`, and `year`.
- Create a class `Library` that:
  - Stores a list of books
  - Has a method `add_book(book_obj)`
  - Has a method `list_books()` that prints all books with title and author

**Test by adding 3 books and listing them.**

---

## SECTION B: NumPy

### Q4. NumPy Array Operations

a. Create a NumPy array of integers from 1 to 12, reshape to 3x4

b. Print the array's:
- Shape, ndim, size
- Mean of entire array
- Min of each row (`axis=1`)
- Index of max element in each column (`axis=0`)

---

### Q5. Special Arrays & Indexing

a. Create the following:
- 3x3 array of zeros
- 4x4 identity matrix
- Array of 5 ones

b. Create an array from 20 to 100 with step 10.

Print:
- First 3 elements
- Last 2 elements
- Reversed array using slicing

---

### Q6. Element-wise Arithmetic

Create two arrays:
```python
a = np.array([10, 20, 30])
b = np.array([1, 2, 3])
```

Perform and print:
- Addition
- Subtraction
- Multiplication
- Division
- Power (`a ** b`)

---

## SECTION C: Pandas

### Q7. Series Operations

a. Create a Pandas Series of 5 temperatures `[30, 35, 32, 33, 31]`
b. Set index labels as weekdays

Print:
- First 3 days
- Max temperature
- Mean temperature

---

### Q8. DataFrame Creation & Info

a. Create a DataFrame with this dictionary:
```python
data = {
    'Name': ['Ali', 'Neha', 'John'],
    'Age': [24, 29, 31],
    'City': ['Delhi', 'Pune', 'Kolkata']
}
```

b. Print:
- Full DataFrame
- Column data types
- Shape and number of rows

---

### Q9. DataFrame Column Operations

- Extend Q8’s DataFrame by adding a new column `Salary = [50000, 60000, 55000]`.
- Then:
  - Increase all salaries by 10%
  - Filter and display rows with salary > 55000
  - Drop the City column

---
