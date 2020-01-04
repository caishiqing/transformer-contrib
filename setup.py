from setuptools import setup, find_packages

setup(name='transformer_contrib',
      version='0.0.1',
      description='Deep learning of NLP',
      author='Cai Shiqing',
      author_email='caishiqing@tom.com',
      url='',
      download_url='',
      license='MIT',
      install_requires=[],
      extras_require={},
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      keywords='NLP, Language Model, Transformer',
      packages=find_packages(),
      package_dir={'transformer_contrib': 'transformer_contrib'},
      )

