## Fixing `winutils.exe` for PySpark on Windows (Spark 4.2.0)

### Environment

* OS: Windows
* Python: virtual environment (`.venv`)
* PySpark: **4.2.0**
* Hadoop (bundled with Spark): **3.5.0**

---

## Problem

Running a Spark script produced this warning:

```text
WARN Shell: Did not find winutils.exe:
java.io.FileNotFoundException:
HADOOP_HOME and hadoop.home.dir are unset.
```

Spark could start, but Hadoop could not find the required Windows utility binary.

---

## Step 1 — Verify Spark and Hadoop versions

Run:

```python
import pyspark
print("PySpark:", pyspark.__version__)

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

print("Spark:", spark.version)
print("Hadoop:", spark.sparkContext._jvm.org.apache.hadoop.util.VersionInfo.getVersion())

spark.stop()
```

Output:

```text
PySpark: 4.2.0
Spark: 4.2.0
Hadoop: 3.5.0
```

This confirmed that Spark was built with **Hadoop 3.5.0**.

---

## Step 2 — Download `winutils.exe`

The `cdarlint/winutils` repository does not provide a Hadoop 3.5.0 build, so I used the latest available stable community build:

* **hadoop-3.3.6**

Download:

* `winutils.exe`

Repository:

* https://github.com/cdarlint/winutils

Path used:

* `hadoop-3.3.6/bin/winutils.exe`

---

## Step 3 — Create the Hadoop directory

Create these folders:

```text
C:\\hadoop
C:\\hadoop\\bin
```

---

## Step 4 — Place the binary

Copy the downloaded file to:

```text
C:\\hadoop\\bin\\winutils.exe
```

Final structure:

```text
C:\\hadoop
└── bin
    └── winutils.exe
```

---

## Step 5 — Set the `HADOOP_HOME` environment variable

Open **System Properties → Advanced → Environment Variables**.

Under **User variables** (or **System variables**), create:

* **Variable name:** `HADOOP_HOME`
* **Variable value:** `C:\\hadoop`

---

## Step 6 — Add Hadoop to PATH

Edit the **Path** variable and add:

```text
C:\\hadoop\\bin
```

---

## Step 7 — Restart the terminal

Close the current Command Prompt / PowerShell window and open a new one so the environment variables are reloaded.

---

## Step 8 — Verify

Run the Spark script again.

The original warning disappeared:

```text
WARN Shell: Did not find winutils.exe
```

New output:

```text
PySpark: 4.2.0
Spark: 4.2.0
Hadoop: 3.5.0
```

A remaining warning may appear:

```text
WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
```

This is **normal on Windows** and does not prevent local PySpark development.

---

## Final working state

* `HADOOP_HOME = C:\\hadoop`
* `PATH` contains `C:\\hadoop\\bin`
* `C:\\hadoop\\bin\\winutils.exe` exists
* PySpark starts successfully
* Spark version: **4.2.0**
* Hadoop version: **3.5.0**
* The `winutils.exe not found` warning is gone

---

## Quick verification command

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

df = spark.range(5)
df.show()

spark.stop()
```

Expected output:

```text
+---+
| id|
+---+
|  0|
|  1|
|  2|
|  3|
|  4|
+---+
```

If this prints successfully, the Windows Spark setup is working correctly.
