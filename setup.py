from setuptools import setup, find_packages

setup(
    name="prompthelix",
    version="0.1.0",
    author="Your Name / Organization",
    author_email="your.email@example.com",
    description="A Python framework for AI prompt generation and optimization using a Prompt DNA System.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/prompthelix", # Replace with your actual URL
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn[standard]",
        "sqlalchemy",
        "psycopg2-binary", # For PostgreSQL
        "pydantic==2.7.1",
        "celery",
        "redis",
        "openai",
        "httpx",
        # "anthropic",
        # "google-generativeai",
    ],
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
