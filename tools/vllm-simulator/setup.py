from setuptools import find_packages, setup


def get_version():
    version = "0.1.0"
    with open("src/vllm_simulator/__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split("=")[1].strip(' \n"')
    return version


setup(
    name="vllm-simulator",
    version=get_version(),
    url="https://github.com/vllm-project/vllm",
    description="A High-Fidelity LLM inference simulator for vLLM",
    packages=find_packages(where="src"),
    package_dir={"": "src"}
)
