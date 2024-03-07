import numpy as np
import tempfile
import os
import subprocess
import sys
from pathlib import Path
from platform import machine
import stat

class MoptaSoftConstraints:
    """
    Mopta08 benchmark with soft constraints as described in https://arxiv.org/pdf/2103.00349.pdf
    Supports i386, x86_84, armv7l

    Args:
        temp_dir: Optional[str]: directory to which to write the input and output files (if not specified, a temporary directory will be created automatically)
        binary_path: Optional[str]: path to the binary, if not specified, the default path will be used
    """

    def __init__(self,):
        self.dims = 124
        self.lb = np.zeros(124, )
        self.ub = np.ones(124, )
        self.sysarch = 64 if sys.maxsize > 2 ** 32 else 32
        self.machine = machine().lower()
        if self.machine == "armv7l":
            assert self.sysarch == 32, "Not supported"
            self._mopta_exectutable = "mopta08_armhf.bin"
        elif self.machine == "x86_64":
            assert self.sysarch == 64, "Not supported"
            self._mopta_exectutable = "mopta08_elf64.bin"
        elif self.machine == "i386":
            assert self.sysarch == 32, "Not supported"
            self._mopta_exectutable = "mopta08_elf32.bin"
        elif self.machine == "amd64":
            assert self.sysarch == 64, "Not supported"
            self._mopta_exectutable = "mopta08_amd64.exe"

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._mopta_exectutable = os.path.join(dir_path, "data", self._mopta_exectutable)
        os.chmod(self._mopta_exectutable, stat.S_IXUSR)

        self.directory_file_descriptor = tempfile.TemporaryDirectory()
        self.directory_name = self.directory_file_descriptor.name

    def __call__(self, x):
        x = np.array(x)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        assert x.ndim == 1
        # write input to file in dir
        with open(os.path.join(self.directory_name, "input.txt"), "w+") as tmp_file:
            for _x in x:
                tmp_file.write(f"{_x}\n")
        # pass directory as working directory to process
        popen = subprocess.Popen(
            self._mopta_exectutable,
            stdout=subprocess.PIPE,
            cwd=self.directory_name,
        )
        popen.wait()
        # read and parse output file
        output = (
            open(os.path.join(self.directory_name, "output.txt"), "r")
            .read()
            .split("\n")
        )
        output = [x.strip() for x in output]
        output = np.array([float(x) for x in output if len(x) > 0])
        value = output[0]
        constraints = output[1:]
        # see https://arxiv.org/pdf/2103.00349.pdf E.7
        return -(value + 10 * np.sum(np.clip(constraints, a_min=0, a_max=None)))



if __name__ == '__main__':
    func = MoptaSoftConstraints()
    for i in range(10):
        x = np.random.rand(124,)
        y = func(x)
        print(y)

