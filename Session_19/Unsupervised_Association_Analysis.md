# Unsupervised Association Analysis: Apriori Algorithm and Frequent Itemsets

## Overview

This notebook covers the concept of **unsupervised association analysis** using the **Apriori algorithm** to find frequent itemsets in transactional data. Association analysis is a fundamental technique in data mining that uncovers relationships and patterns among items in large datasets, often applied in market basket analysis, web usage mining, and bioinformatics.

---

## Step-by-Step Explanation

### 1. What is Association Analysis?

Association analysis aims to discover interesting relationships (associations or correlations) among variables in large databases. The most common example is **market basket analysis**, where we analyze customer purchases to find items that are frequently bought together.

- **Example:** If customers often buy milk and bread together, a retailer might promote these items jointly.

---

### 2. The Dataset

The dataset used in this notebook is a small, illustrative transactional dataset:

```python
dataset = [
    ['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
    ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
    ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
    ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
    ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Eggs', 'Ice cream']
]
```

Each sublist represents a "transaction" (e.g., a shopping basket), with the items purchased.

---

### 3. Data Preprocessing

**Goal:** Convert the dataset into a format suitable for association rule mining.

- **TransactionEncoder:** Part of the `mlxtend` library, it transforms the list of transactions into a one-hot encoded boolean DataFrame.

```python
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_items = te.fit(dataset)
print(te_items.columns_)  # Outputs all unique items in the dataset
df = pd.DataFrame(te.transform(dataset), columns=te_items.columns_)
```

**Resulting DataFrame:**

| Apple | Corn | Dill | Eggs | Ice cream | Kidney Beans | Milk | Nutmeg | Onion | Unicorn | Yogurt |
|-------|------|------|------|-----------|--------------|------|--------|-------|---------|--------|
| False | False| False| True | False     | True         | True | True   | True  | False   | True   |
| False | False| True | True | False     | True         | False| True   | True  | False   | True   |
| True  | False| False| True | False     | True         | True | False  | False | False   | False  |
| False | True | False| False| False     | True         | True | False  | False | True    | True   |
| False | True | False| True | True      | True         | False| False  | True  | False   | False  |

Each row is a transaction; each column is an item; each cell is True if the item is present in the transaction.

---

### 4. Finding Frequent Itemsets with Apriori

The **Apriori algorithm** identifies itemsets that appear frequently across transactions, based on a minimum support threshold.

- **Support:** The proportion of transactions that contain the itemset.

```python
from mlxtend.frequent_patterns import apriori

frequent_itemsets = apriori(df, use_colnames=True, min_support=0.4)
```

- `min_support=0.4` means the itemset must appear in at least 40% of transactions (i.e., at least 2 out of 5 transactions).

**Sample Output Table:**

| support | itemsets                                  |
|---------|-------------------------------------------|
| 0.4     | (Corn)                                    |
| 0.8     | (Eggs)                                    |
| 1.0     | (Kidney Beans)                            |
| 0.6     | (Milk)                                    |
| 0.4     | (Nutmeg)                                  |
| 0.6     | (Onion)                                   |
| 0.6     | (Yogurt)                                  |
| 0.8     | (Eggs, Kidney Beans)                      |
| 0.6     | (Eggs, Onion)                             |
| 0.6     | (Kidney Beans, Milk)                      |
| ...     | ...                                       |

This table shows itemsets and their support values.

---

### 5. Examples and Interpretation

- **Single Items:**  
  - `Kidney Beans` have support 1.0 (appear in all transactions).
  - `Eggs` have support 0.8 (appear in 4 out of 5 transactions).

- **Pair Itemsets:**  
  - `(Eggs, Kidney Beans)` have support 0.8 (appear together in 4 transactions).
  - `(Milk, Yogurt)` have support 0.4 (appear together in 2 transactions).

- **Larger Itemsets:**  
  - `(Eggs, Kidney Beans, Milk)` have support 0.4 (appear together in 2 transactions).
  - `(Nutmeg, Onion, Eggs, Yogurt, Kidney Beans)` have support 0.4 (appear together in 2 transactions).

The algorithm can find associations like:
- If a customer buys `Kidney Beans`, they are likely to buy `Eggs`.
- `Milk` and `Yogurt` are frequently bought together.

---

## Why is This Useful?

- **Market Basket Analysis:** Retailers can use these insights to arrange store layouts, create combo offers, and optimize inventory.
- **Recommendation Systems:** Online stores can recommend items commonly bought together.
- **Healthcare:** Identifying co-occurring symptoms or treatments.

---

## Summary

- **Association analysis** extracts patterns of co-occurrence from transaction data.
- The **Apriori algorithm** finds frequent itemsets using a support threshold.
- **Frequent itemsets** help uncover valuable combinations of items.

---

## References

- [mlxtend.frequent_patterns.apriori Documentation](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/)
- [Association Rule Mining — Market Basket Analysis](https://en.wikipedia.org/wiki/Association_rule_learning)
