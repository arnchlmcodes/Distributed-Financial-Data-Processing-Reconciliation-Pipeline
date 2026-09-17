"""
conftest.py — Global pytest configuration and fixtures.
Ensures PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON are set to sys.executable.
"""
import os
import sys

# Ensure Spark workers always find the current Python virtualenv
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["PYTHONPATH"] = os.path.abspath(os.path.dirname(__file__))
if os.path.exists(r"C:\hadoop\bin\winutils.exe"):
    os.environ["HADOOP_HOME"] = r"C:\hadoop"
    os.environ["hadoop.home.dir"] = r"C:\hadoop"
    os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ.get("PATH", "")
