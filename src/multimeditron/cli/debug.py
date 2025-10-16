from multimeditron.cli import EPILOG, CONFIG_PATH, main_cli
from multimeditron.utils import get_torch_dtype
from datasets import load_dataset
import ray
from ray import serve
from fastapi import Request


@serve.deployment(num_replicas=2)  # scale horizontally if needed
class PyExecService:
    def __init__(self):
        # create a single NsJailExecutor actor for each replica
        # self.executor = NsJailExecutor.remote()
        pass

    async def __call__(self, request: Request):
        """
        HTTP handler:
        - expects POST with JSON body {"code": "print('hello')", "timeout": 5}
        - runs code in nsjail
        - returns JSON result
        """
        data = await request.json()
        code = data.get("code", "")
        # timeout = data.get("timeout", 5)

        if not code.strip():
            return {"error": "No code provided"}

        # execute asynchronously via Ray
        # result = await self.executor.execute.remote(code, wall_timeout=timeout)

        return {"not": "implemented"}

@main_cli.command("serve")
def _serve():
    # Start ray if not already running
    ray.init(address="auto", namespace="serve")

    # Deploy service
    app = PyExecService.bind()
    serve.run(app, blocking=True)

    print("🚀 Ray Serve running at http://127.0.0.1:8000/PyExecService")
