from stuptools import setup, find_packages

setup(
    name='memorynetwork',
    description='',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'nltk',
        'keras',
    ],
)
