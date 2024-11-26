from setuptools import setup, find_packages

setup(
	name='project2',
	version='1.0',
	author='Aashritha Reddy Donapati',
	author_email='a.donapati@ufl.edu',
	packages=find_packages(exclude=('tests', 'docs', 'resources')),
	setup_requires=['pytest-runner'],
	tests_require=['pytest']	
)
