import setuptools

if __name__ == "__main__":
    setuptools.setup( name="nnUnet",
    version="0.1",
    packages=setuptools.find_packages(include=['nnunetv2', 'monai'])
    )
