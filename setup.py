
from setuptools import find_packages, setup



__name__ = ""
__author__ = ""
__author_email__ = ""
__license__ = ""
__url__ = ""
__description__ = "description"

#python -m spacy download en_core_web_lg

setup(
    name = __name__,
    version = __version__,    
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = find_packages(),
    install_requires="",

    classifiers=[
                'Development Status :: 4 - Beta',
                'Programming Language :: Python :: 3.7',
                ],
)

