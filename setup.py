"""
Setup script for Weather-UPI Dashboard
"""

from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="weather-upi-dashboard",
    version="1.0.0",
    description="Weather-UPI Correlation Dashboard with MCP Integration",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "weather-upi-pipeline=main:main",
        ],
    },
)