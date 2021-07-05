import os
from setuptools import setup, find_packages

with open('VERSION', 'r') as f:
    version = f.read().strip()
    if version.endswith("dev"):
        version = version[:-3]


def req_file(filename, folder="requirements"):
    with open(os.path.join(folder, filename)) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters
    # Example: `\n` at the end of each line
    return [x.strip() for x in content]


install_requires = req_file("requirements.txt")


setup(
    name='marveltoolbox',
    version=version,
    packages=find_packages(),
    author="Renjie Xie",
    author_email="renjie_xie@seu.edu.cn",
    maintainer="Renjie Xie",
    maintainer_email="renjie_xie@seu.edu.cn",
    url="https://github.com/xrj-com/marveltoolbox",
    download_url="https://github.com/xrj-com/marveltoolbox",
    license="GNU GPLv3 License",
    description='A toolbox for DL communication research.',
    classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Information Technology',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Utilities',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Environment :: Console',
            'Natural Language :: English',
            'Operating System :: OS Independent',
    ],
    keywords='deep learning, ' \
             'machine learning, supervised learning, ' \
             'unsupervised learning, communication, ' \
             'complex value matrix computiton', 
    python_requires='>=3.8',
    platforms=["Linux"],
    # data_files=[('',['VERSION'])],
    include_package_data=True,
    install_requires=install_requires,
)