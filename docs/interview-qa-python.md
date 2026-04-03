# Interview Q&A — Python

*Core language topics: semantics, data model, concurrency, and practical patterns commonly asked in backend and ML engineering interviews.*

**How to use these answers:** Each **Answer** starts with an **easy-to-follow core**; **If they want more:** adds precise terms and edge cases—same pattern as [`interview-qa-rag-senior.md`](interview-qa-rag-senior.md).

---

## Language fundamentals

### 1. What is the difference between `is` and `==`?

**Answer:** Use **`==`** when you care whether **two values are equal** (same content for numbers, strings, etc.). Use **`is`** only when you care whether **two names point to the exact same object in memory**. For everyday equality checks, **`==`** is the right default.

**If they want more:** `==` calls `__eq__`. `is` checks object identity. For `None`, the idiomatic check is `x is None`. Small integers and interned strings can make `is` look like it works for equality—**don’t rely on that**; it is implementation-dependent.

---

### 2. What are mutable vs immutable types?

**Answer:** **Immutable** values cannot be changed in place—you make a **new** object when you “change” them (typical: numbers, strings, tuples of immutables). **Mutable** values can be updated **without** replacing the object (lists, dicts, most custom classes). This matters when you **share** objects between functions or threads: changing a shared **mutable** affects everyone holding a reference.

**If they want more:** Immutables can be **dict keys** if hashable; mutables usually cannot. Default arguments and **copying** (shallow vs deep) behave differently depending on mutability.

---

### 3. Why should you avoid mutable default arguments?

**Answer:** Default values are created **once**, when Python **defines** the function—not on every call. So a list default is **the same list every time**; one caller appends, and the next caller sees the leftovers.

**If they want more:** Fix with `def f(items=None): items = [] if items is None else items` (or a tuple default if appropriate).

---

### 4. What is a generator and why use one?

**Answer:** A **generator** produces values **one at a time** instead of building a huge list in memory. You often write them with **`yield`** or a generator expression `(x for x in ...)`. They are great for **large streams** or **pipelines** where you do not need all items at once.

**If they want more:** Generators are **single-pass** unless you re-run them or use `itertools.tee`. Memory stays small because only current state is held.

---

### 5. What is the difference between `*args` and `**kwargs`?

**Answer:** **`*args`** gathers extra **positional** arguments into a **tuple**. **`**kwargs`** gathers extra **keyword** arguments into a **dict**. They let you write flexible functions and **forward** arguments to another function.

**If they want more:** Common pattern: `def wrapper(*args, **kwargs): return inner(*args, **kwargs)`.

---

## Data model and OOP

### 6. Explain `__init__` vs `__new__`.

**Answer:** **`__new__`** actually **creates** the object (it runs first). **`__init__`** **fills in** the object after it exists. In day-to-day code you almost always only implement **`__init__`**.

**If they want more:** Override **`__new__`** for things like **singletons**, subclassing **immutable** types, or custom allocation. Normal classes: `__new__` allocates, `__init__` initializes.

---

### 7. What is the MRO (Method Resolution Order)?

**Answer:** When you call a method on an object, Python must decide **which class’s version** to run, especially with **multiple inheritance**. The **MRO** is that **linear search order** through the base classes.

**If they want more:** CPython uses **C3 linearization**. Inspect with `ClassName.__mro__`. It is what makes cooperative `super()` work in complex hierarchies.

---

### 8. What are `@property` and setters used for?

**Answer:** They let you expose **computed** or **validated** values using **attribute syntax** (`obj.x`) while still running code behind the scenes. Callers do not need to switch to `get_x()` / `set_x()` when you add logic later.

**If they want more:** `@property` for reads; `@name.setter` for controlled writes. Useful for **backward-compatible** API evolution.

---

### 9. What is a descriptor?

**Answer:** A **descriptor** is an object with `__get__`, `__set__`, or `__delete__` that controls what happens when you access an attribute on a **class** instance. It is Python’s built-in hook for **custom attribute behavior**.

**If they want more:** `property`, `classmethod`, and `staticmethod` are implemented with descriptors. They are the low-level mechanism under attribute access.

---

## Internals and concurrency

### 10. What is the GIL?

**Answer:** In **CPython**, the **Global Interpreter Lock** means **only one thread runs Python bytecode at a time**. So multiple threads do not give you **parallel CPU work** on pure Python the way multiple processes do.

**If they want more:** **I/O-bound** work can still overlap (threads release the GIL when waiting). **CPU-bound** pure Python often uses **`multiprocessing`**, native extensions, or other runtimes. The GIL simplifies memory management in the main interpreter.

---

### 11. When would you use `asyncio` vs threads vs processes?

**Answer:** **`asyncio`**: lots of **waiting** on networks or disks, one thread, explicit **`async`/`await`**—good when libraries support it. **Threads**: simpler for **blocking** I/O code that will not block *everyone* if the GIL is released. **Processes**: **true parallelism** for heavy **CPU** Python on multiple cores, with higher **startup and IPC** cost.

**If they want more:** Profile first; `async` does not speed up CPU-bound numpy/sklearn by itself.

---

### 12. What is the difference between `multiprocessing` and `threading` for CPU-heavy Python code?

**Answer:** For **CPU-heavy pure Python**, **threads** still take turns because of the GIL, so you do not get parallel speedup. **Separate processes** each have their own Python interpreter and **can use multiple cores**, at the cost of more memory and pickling/IPC.

---

## Practical patterns and ecosystem

### 13. How do virtual environments help?

**Answer:** They give each project its **own installed packages**, so one project’s **versions** do not break another’s. You avoid “it worked on my machine” clashes with the system Python.

**If they want more:** Typical tools: **`venv`** (stdlib), **Poetry**, **pip-tools**, **uv** for reproducible lockfiles.

---

### 14. What is the difference between shallow and deep copy?

**Answer:** **Shallow copy** makes a **new container** but **inner** mutable objects are still **shared** with the original. **Deep copy** duplicates **nested** structures so nested lists/dicts are independent.

**If they want more:** Use `copy.copy` vs `copy.deepcopy`. Pick based on whether nested objects must be **isolated**.

---

### 15. How does exception handling work? What are `else` and `finally`?

**Answer:** **`try`** runs code that might fail. **`except`** catches specific errors—prefer **named** exception types over catching everything. **`else`** runs only if **`try` completed without exception**. **`finally`** runs **always**—use it for cleanup (close files, release locks) whether or not an error happened.

**If they want more:** Avoid bare `except:` that hides **`KeyboardInterrupt`** unless you truly intend to.

---

### 16. What are type hints used for?

**Answer:** They are **documentation** and **checker input**: you write `x: int` so humans and tools like **mypy** know what you meant. Python **does not** enforce them at runtime by default.

**If they want more:** **Pydantic** and similar libraries add **runtime validation** where needed (e.g. API boundaries).

---

### 17. What is a context manager and the `with` statement?

**Answer:** **`with`** guarantees **cleanup** after a block runs—even if an exception happens. Typical use: **open a file** and always **close** it without writing `try`/`finally` by hand every time.

**If they want more:** Context managers implement `__enter__` / `__exit__` or use `@contextmanager`. Good for files, locks, DB transactions.

---

## Idioms and “gotchas”

### 18. What does `if __name__ == "__main__":` do?

**Answer:** Code under this guard runs when the file is executed as a **script** (`python file.py`), not when it is **imported** as a module. Use it for **CLI entry points** so `import` does not accidentally run your whole program.

---

### 19. List comprehension vs `map` / `filter`?

**Answer:** **List comprehensions** are usually **easier to read** in Python for building lists with filters or nested logic. **`map`/`filter`** return **iterators** in Python 3. Pick **readability**; speed is rarely the deciding factor.

---

### 20. What is iterable unpacking with `*`?

**Answer:** **`*`** “spreads” an iterable into separate items. In a function call, `f(*items)` passes each element as its **own argument**. In assignment, `a, *rest = items` captures **the rest** into a list. **`**`** does the same for **dict** keyword merging and `**kwargs`.

---
