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
    name="sns",
    version="2.0.0",
    author="SNS Development Team",
    author_email="your.email@example.com",
    description="SNS: Self-Nonself Modeling for Automatic Survey Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yurui12138/IG-Finder",
    license="MIT License",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: General",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'sns=knowledge_storm.sns.cli:main',
        ],
    },
    keywords=[
        'self-nonself modeling',
        'cognitive lag',
        'automatic survey',
        'taxonomy evolution',
        'scientific research',
        'multi-view baseline',
        'stress test',
        'delta-aware guidance',
        'literature analysis',
    ],
)
