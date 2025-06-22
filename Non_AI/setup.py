from setuptools import setup, find_packages

setup(
    name="Non_AI",  # Replace with your project's name
    version="0.1.0",  # Start with a basic version number
    packages=find_packages(where="."),  # Automatically find your packages
    package_dir={"": "."},
    install_requires=[
        "opencv-python",
        "numpy",
        "scikit-image",
        "imutils",
        "scikit-learn",
        "scipy"
    ],  # Add your project's dependencies
)