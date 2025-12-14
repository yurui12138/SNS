import re

from setuptools import setup, find_packages

# Read the content of the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()
    # Remove p tags.
    pattern = re.compile(r"<p.*?>.*?</p>", re.DOTALL)
    long_description = re.sub(pattern, "", long_description)

# Read the content of the requirements.txt file
with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()


setup(
    name="ig-finder",
    version="0.1.0",
    author="IG-Finder Development Team",
    author_email="your.email@example.com",
    description="IG-Finder: Innovation Gap Identification Framework for Scientific Research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yurui12138/storm",
    license="MIT License",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'ig-finder=knowledge_storm.ig_finder.cli:main',
        ],
    },
    keywords=[
        'innovation gap',
        'scientific research',
        'knowledge curation',
        'cognitive baseline',
        'literature analysis',
        'ai research assistant',
    ],
)
