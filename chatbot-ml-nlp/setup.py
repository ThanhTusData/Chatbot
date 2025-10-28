from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="chatbot-ml-nlp",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced ML/NLP Chatbot with Intent Classification and Semantic Retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/chatbot-ml-nlp",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "test": ["pytest", "pytest-cov", "pytest-asyncio"],
    },
    entry_points={
        "console_scripts": [
            "chatbot-serve=serving.fastapi_app:main",
            "chatbot-train=training.train_intent:main",
            "chatbot-build-kb=scripts.build_kb_index:main",
        ],
    },
)