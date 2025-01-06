# Running Stable-Audio-Tools with Apolo

This guide describes how to set up and run various jobs (training, inference, model unwrapping, and parameter grid search) on the Apolo platform. We'll assume you have Apolo CLI installed and are familiar with basic Apolo workflows.

## Prerequisites

1. **Clone the Repository**  
   Ensure you are on the `MLO-71` branch:
   ```bash
   git clone https://github.com/neuro-inc/stable-audio-tools.git
   cd stable-audio-tools
   git checkout MLO-71
   ```
2.  **Set Up Weights & Biases API Key**  
    Before running any training or job that logs metrics, you'll need to set a secret for W&B.  
    Replace `<YOUR_WANDB_API_KEY>` with your actual W&B key:
    
    ```bash
    apolo secret add WANDB_API_KEY <YOUR_WANDB_API_KEY>
    ```
    
3.  **Data & Checkpoints**  
    You can organize your dataset and checkpoint files locally at the root level of the repo, as shown:
    
    ```bash
    stable-audio-tools/ 
     ├── .apolo/ 
     │   ├── live.yaml 
     │   ├── matrix.yaml 
     │   └── project.yaml 
     ├── checkpoints/ 
     │   └── model-001.ckpt 
     ├── dataset/ 
     │   ├── my_dataset/... 
     │   └── training_metadata.csv 
     ├── ...
    ```

    *   Ensure `dataset/` contains your audio data and `training_metadata.csv`.
    *   Optionally, place a `model-001.ckpt` checkpoint in `checkpoints/` if you have a pretrained model.
    
    Uou can upload these directories to Apolo storage manually with the below commands or run a inference or training job that will upload them for you:
    
    ```
    apolo-flow upload dataset 
    apolo-flow upload checkpoints
    ```
    
    This makes the dataset and checkpoints available for all jobs (training, inference, etc.). If you prefer, you can manage storage differently (e.g., upload at runtime or store files directly on Apolo storage), but the above approach is recommended for simplicity.
    

Building the Image
------------------

The `live.yaml` and `matrix.yaml` files reference an image built from the provided `Dockerfile`. To build the image:

```
apolo-flow build stable_audio_tools
```

This command will:

*   Use the Dockerfile in the root directory.
*   Build and push the image to the Apolo registry.
*   The image will then be referenceable by `image:$[[ project.id ]]:v1`.

Running the Training Job (Live Workflow)
----------------------------------------

The live workflow is defined in `.apolo/live.yaml`. After building the image, you can run the training job defined there:

```
apolo-flow run training
```

**What it Does:**

*   Uses the mounted `dataset` and `checkpoints` volumes (uploaded earlier).
*   Runs `train.py` with parameters specified in the workflow file.
*   Logs training metrics to W&B.
*   Saves checkpoints to `/app/checkpoints` on Apolo storage.

By default, this will run a single-GPU training job for `my_experiment`. It will store checkpoints under `/app/checkpoints` and can be monitored via W&B and Apolo CLI logs.

Running the Inference Job (Live Workflow)
-----------------------------------------

After you have a trained or unwrapped model checkpoint, you can run the inference Gradio UI job:

```
apolo-flow run inference
```

**What it Does:**

*   Launches a Gradio interface to interact with the model.
*   Serves on an HTTP endpoint protected by Apolo.
*   Allows you to test the model's generation capabilities.

You can see the URL for the Gradio interface by checking `apolo-flow status inference`.

Unwrapping the Model (Live Workflow)
------------------------------------

Once training is complete, you often need to unwrap the model checkpoint for inference. This removes training wrappers and leaves you with a clean checkpoint. To do this:

1.  Identify your final training checkpoint (`<my_model_name>/<run_id>/checkpoints/epoch=...ckpt`).
2.  Run the unwrapping job:
    
    
    ```
    apolo-flow run unwrap
    ```
    

**What it Does:**

*   Runs `unwrap_model.py` with the provided checkpoint.
*   Produces a standalone `.ckpt` file suitable for direct inference or fine-tuning.

After completion, you'll find `unwrapped_model.ckpt` in the `/app/checkpoints` directory on Apolo storage.

Running the Grid Search (Batch Workflow)
----------------------------------------

The `.apolo/matrix.yaml` defines a batch workflow for a parameter grid search. This allows running multiple training configurations in parallel, exploring different hyperparameters.

Before starting the batch:

*   Ensure `WANDB_API_KEY` secret is set.
*   Ensure dataset and checkpoints are uploaded (if not, run `apolo-flow upload dataset` and `apolo-flow upload checkpoints` again).
*   The batch workflow pulls configurations from `generate_config.py` and `train.py`.

Run the batch:

```
apolo-flow bake matrix
```

**What it Does:**

*   Executes multiple training tasks in parallel (limited by `max_parallel` in `matrix.yaml`).
*   Varies learning rate, batch size, weight decay, encoder latent dim, diffusion depth, etc.
*   Logs each run to W&B under a specified experiment name.
*   Results and checkpoints appear in `/app/checkpoints`.

**After Completion:** You can use W&B to compare metrics across runs or check Apolo logs for each job to see training progress.

Summary
-------

*   **Set W&B Secret**: `apolo secret add WANDB_API_KEY <YOUR_KEY>`
*   **Build Image**: `apolo-flow build stable_audio_tools`
*   **Upload Local Data/Checkpoints**: `apolo-flow upload dataset` and `apolo-flow upload checkpoints`
*   **Run Training**: `apolo-flow run training`
*   **Run Inference UI**: `apolo-flow run inference`
*   **Unwrap Model**: `apolo-flow run unwrap`
*   **Run Grid Search**: `apolo-flow bake matrix`

These steps let you manage the full lifecycle of training, testing, and optimizing your stable-audio-tools model on Apolo. You can interact with storage as you see fit—either pre-uploading via `apolo-flow upload` or directly referencing remote storage paths.
