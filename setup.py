import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="or_models",
    version="0.1.0",
    url="https://github.com/hak91/OptimizationModels",
    author="Hussain Kharodawala",
    author_email="hussain.abbas52@gmail.com",
    description="Some standard textbook Optimization Models and Algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    keywords = ['set cover', 'Operations Research','Mathematical Optimization', 'heuristic', 'greedy', 'Lagrangian Relaxation'],
    install_requires=[
        'numpy',
        'time',
        'ortools'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.8'
    ],
)