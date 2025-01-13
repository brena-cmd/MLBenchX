from setuptools import setup, find_packages

setup(
    name='mlbenchx',  # Nome do pacote
    version='0.1',
    packages=find_packages(),  # Encontra os pacotes automaticamente
    install_requires=[
        'numpy',
        'scikit-learn',
        'pandas',
        'numpy',
        'joblib',
        'matplotlib',
        'seaborn',
        'argparse'# Exemplo de dependência
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',  # Versão mínima do Python
)
