# Multimeditron Base Chat 🩺

A lightweight Gradio-based web UI to **test** and **chat** with our Multimeditron models (e.g., `MultiMeditron-8B-CLIP`).


## Directory Layout

```
ui/
├── app.py                 # Main Gradio app (UI + inference glue)
├── assets/
│   ├── chat.css           # UI layout & overflow fixes
│   └── Meditron8B_Logo_with_cube.png  # Logo
└── launch_gradio.sh      # convenience launcher for the Gradio UI (sets PORT/env vars and runs app.py)
```

## Usage

```bash
# On the cluster:  Connect to node - interactive job
srun --time=1:29:59 --partition normal -A a127 --environment=/users/<cscs-username>/.edf/multimodal.toml --pty bash


# on your local machine run
# in ~/.ssh/config : add 
Host debug
    HostName $NODEID # replace with the actual node id from your interactive job e.g nid007648
    User $CSCS-USERNAME # replace with your cscs username
    ProxyJump clariden
    ForwardAgent yes
    ForwardX11 yes
    
    
# on your local machine
ssh -N -L $PORT:localhost:$PORT debug  # put the port defined in launch_gradio.sh
```

Then, on the cluster interactive node, run:

```bash
./launch_gradio.sh
```

Open your browser locally at: `http://localhost:$PORT`.


## Troubleshooting 
So that you don't replicate my errors...

* **Blank page / cannot connect**: ensure your SSH tunnel uses the **same port** as `--server_port`, and that the app binds to `0.0.0.0`.
* **Model load errors**: check that your checkpoint folder contains `model.safetensors.index.json` and the corresponding shard files and that tokenizer files are present (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`).

