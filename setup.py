"""
AWESEOME EDA API to make EDA <-> Modeling process easier

@author: Harsh Desai
@url: https://github.com/hurshd0/awesome-eda/
"""

# Always prefer setuptools over distutils
from setuptools import setup
import os

readme_path = os.path.join('README.md')
with open(readme_path, encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='awesome-eda',
    version = '0.0',
	packages=['awesome_eda'],
    description='Awesome EDA API to make EDA <-> Modeling process easier',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author = 'Harsh Desai',
    author_email = 'hurshd0@gmail.com',
    license='MIT',
    url='https://github.com/hurshd0/awesome-eda/',
	classifiers = [],
	install_requires=[
          'pandas<=0.23.4',
		  'matplotlib<=3.1.0',
		  'plotly'>='4.0',
          'numpy',
          'scipy',
		  'scikit-learn',
		  'seaborn',
		  'category_encoders',
		  'statsmodel',
		  'shap',
		  'pdpbox',
		  'xgboost'
      ],
)