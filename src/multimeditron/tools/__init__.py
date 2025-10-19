from omegaconf import OmegaConf
from typing import Optional, Tuple
import tempfile
import random
import shutil
import ray
import os


@ray.remote
class NsJailExecutor:
    def __init__(self, cfg):
        self.nsjail_path = cfg.nsjail.path
        self.python_path = cfg.python.path

        self.default_rlimit_as = cfg.nsjail.get("default_rlimit_as", 256 * 1024 * 1024)  # 256MB
        self.default_rlimit_cpu = cfg.nsjail.get("default_rlimit_cpu", 2)  # 2 seconds of CPU time
        self.default_time_limit = cfg.nsjail.get("default_time_limit", 5) # 5 seconds of wall time
        self.default_open_fds = cfg.nsjail.get("default_open_fds", 16)  # max number of open file descriptors is 16

        self.allow_network = cfg.nsjail.get("allow_network", False)

    def _build_nsjail_cmd(self,
                          workdir: str,
                          code_filename: str,
                          rlimit_as: Optional[int] = None,
                          rlimit_cpu: Optional[int] = None,
                          time_limit: Optional[int] = None,
                          open_fds: Optional[int] = None) -> list:
        """
        Build nsjail CLI args; You may need to adapt bindmount/chroot paths based on your environment. This
        uses a conservative set of flags most nsjail builds support.
        """
        # Use provided or default values
        rlimit_as = rlimit_as if rlimit_as is not None else self.default_rlimit_as
        rlimit_cpu = rlimit_cpu if rlimit_cpu is not None else self.default_rlimit_cpu
        time_limit = time_limit if time_limit is not None else self.default_time_limit
        open_fds = open_fds if open_fds is not None else self.default_open_fds

        # Example flags:
        #  --chroot: we set chroot to the workdir (empty dir)
        #  --user/--group: drop priveleges inside the jail (use nobody/nogroup uid/gid)
        #  --disable_proc: avoid mounting /proc inside the jail (safer)
        #  --rlimit_as / --rlimit_cpu: limit memory and CPU time
        #  --time_limit: limit wall time (in seconds)
        #  --max_fsize: limit max filesystem size (in bytes)
        #  --cwd: set working directory inside the jail (we set to / which is the chroot)
        #  --   : end of nsjail args, beginning of command to run inside the jail
        uid = 65534  # nobody
        gid = 65534  # nogroup

        cmd = [
            self.nsjail_path,
            "--mode", "o", # "once" mode
            "--chroot", workdir,
            "--cwd", "/",
            "--user", str(uid),
            "--group", str(gid),
            "--disable_proc",
            "--rlimit_as", str(rlimit_as),
            "--rlimit_cpu", str(rlimit_cpu),
            "--time_limit", str(time_limit),
            "--max_fsize", str(5 * 1024 * 1024),  # 5MB
            "--rlimit_nofile", str(open_fds),
            "--keep_caps", # keep capabilities false? nsjail may drop them anyway
        ]

        # Bind-mount the code file into the chroot so python inside the jail can access it.
        # We mount it at /code.py inside the chroot.
        src_code_path = os.path.join(workdir, code_filename)  # absolute on host
        cmd += ["--bindmount", f"{src_code_path}:/code.py:ro"]

        # Bind-mount the python interpreter (if needed). If interpreter is available inside chroot, skip this.
        # Note: bind-mounting interpreter may not be sufficient if shared libs are needed (more bind mounts required).
        if os.path.exists(self.python_interpreter):
            cmd += ["--bindmount", f"{self.python_interpreter}:{self.python_interpreter}:ro"]

        # As final part, the program to execute inside the jail
        cmd += ["--", self.python_interpreter, "/code.py"]

        return cmd
    
    @staticmethod
    def _prepare_workdir(self, user_code: str) -> Tuple[str, str]:
        """
        Create a temporary directory and write code to a file.
        We will chroot into that directory. Keep it minimal.
        Returns (workdir_abs_path, filename).
        """
        tmpdir = tempfile.mkdtemp(prefix="nsjail_exec_")
        # make a code filename with safe name
        code_filename = "code.py"
        code_path = os.path.join(tmpdir, code_filename)
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(user_code)
        # ensure minimal permissions
        os.chmod(code_path, 0o444)  # read-only r--r--r--
        return tmpdir, code_filename

    @staticmethod
    def _cleanup_workdir(path: str):
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass

    @staticmethod
    def _ensure_path_executable(path: str):
        if not os.path.isfile(path) or not os.access(path, os.X_OK):
            raise FileNotFoundError(f"The path '{path}' is not an executable file or is not accessible. Install it and ensure it is accessable from the worker nodes.")

