from pathlib import Path
from setuptools import setup, find_packages


def parse_requirements(path: str) -> list[str]:
    """Read requirements from the given file."""
    lines = Path(path).read_text().splitlines()
    reqs = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(line)
    return reqs

setup(
    name="prompthelix",
    version="0.1.0",
    author="Your Name / Organization",
    author_email="your.email@example.com",
    description="A Python framework for AI prompt generation and optimization using a Prompt DNA System.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/prompthelix", # Replace with your actual URL
    license="MIT",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Development Status :: 3 - Alpha", # Or "4 - Beta", "5 - Production/Stable"
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "License :: OSI Approved :: MIT License", # Choose your license
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires='>=3.9',
    entry_points={
        "console_scripts": [
            "prompthelix=prompthelix.cli:main_cli",
        ],
    },
)
