import json
import os
from hashlib import md5
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F
from functorch import grad, make_functional_with_buffers, vmap
from peft import PeftModel
from torch import Tensor
from torch.nn.functional import normalize
from tqdm import tqdm
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
from transformers import RobertaModel


def prepare_batch(batch, device=None):
    """ Move the batch to the device. """
    if device is None:
        # Auto-detect device from batch if available, otherwise use cuda:0
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    
    for key in batch:
        batch[key] = batch[key].to(device)
    return batch  # Return the modified batch


def get_max_saved_index(output_dir: str, prefix="reps") -> int:
    """ 
    Retrieve the highest index for which the data (either representation or gradients) has been stored. 

    Args:
        output_dir (str): The output directory.
        prefix (str, optional): The prefix of the files, [reps | grads]. Defaults to "reps".

    Returns:
        int: The maximum representation index, or -1 if no index is found.
    """

    files = [file for file in os.listdir(
        output_dir) if file.startswith(prefix)]
    index = [int(file.split(".")[0].split("-")[1])
             for file in files]  # e.g., output_dir/reps-100.pt
    return max(index) if len(index) > 0 else -1


def get_output(model,
               weights: Iterable[Tensor],
               buffers: Iterable[Tensor],
               input_ids=None,
               attention_mask=None,
               labels=None,
               ) -> Tensor:
    logits = model(weights, buffers, *(input_ids.unsqueeze(0),
                   attention_mask.unsqueeze(0))).logits
    labels = labels.unsqueeze(0)
    loss_fct = F.cross_entropy
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
    return loss


def get_trak_projector(device: torch.device):
    """ Get trak projectors (see https://github.com/MadryLab/trak for details) """
    try:
        num_sms = torch.cuda.get_device_properties(
            device.index).multi_processor_count
        import fast_jl

        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(torch.zeros(
            8, 1_000, device=device), 512, 0, num_sms)
        projector = CudaProjector
    except:
        projector = BasicProjector
    return projector


def get_number_of_params(model):
    """ Make sure that only lora parameters require gradients in peft models. """
    if isinstance(model, PeftModel):
        names = [n for n, p in model.named_parameters(
        ) if p.requires_grad and "lora" not in n]
        assert len(names) == 0
    num_params = sum([p.numel()
                     for p in model.parameters() if p.requires_grad])
    return num_params


def obtain_gradients(model, batch):
    """ obtain gradients. """
    loss = model(**batch).loss
    loss.backward()
    vectorized_grads = torch.cat(
        [p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    return vectorized_grads


def obtain_sign_gradients(model, batch):
    """ obtain gradients with sign. """
    loss = model(**batch).loss
    loss.backward()

    # Instead of concatenating the gradients, concatenate their signs
    vectorized_grad_signs = torch.cat(
        [torch.sign(p.grad).view(-1) for p in model.parameters() if p.grad is not None])

    return vectorized_grad_signs


def obtain_gradients_with_adam(model, batch, avg, avg_sq):
    """ obtain gradients with adam optimizer states. """
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-08

    loss = model(**batch).loss
    loss.backward()

    # Collect gradients and move them to the same device (for multi-GPU models)
    # Use the device of the first gradient as the target device
    grads_list = []
    target_device = None
    for n, p in model.named_parameters():
        if p.grad is not None:
            if target_device is None:
                target_device = p.grad.device
            grads_list.append(p.grad.view(-1).to(target_device))
    vectorized_grads = torch.cat(grads_list)

    # Move avg and avg_sq to the same device as vectorized_grads
    avg = avg.to(target_device)
    avg_sq = avg_sq.to(target_device)
    
    updated_avg = beta1 * avg + (1 - beta1) * vectorized_grads
    updated_avg_sq = beta2 * avg_sq + (1 - beta2) * vectorized_grads ** 2
    vectorized_grads = updated_avg / torch.sqrt(updated_avg_sq + eps)

    return vectorized_grads


def prepare_optimizer_state(model, optimizer_state, device):
    # Optimizer state keys are like: base_model.model.model.layers.0.self_attn.k_proj.lora_A.default.weight
    # Model parameter names from PeftModel are like: base_model.model.model.layers.0.self_attn.k_proj.lora_A.default.weight
    # We need to match them correctly
    model_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
    
    # Try to match optimizer state keys with model parameter names
    # First, try direct match
    matched_names = []
    for model_name in model_param_names:
        # Try different prefixes to match optimizer state keys
        possible_keys = [
            model_name,  # Direct match
            'base_model.model.' + model_name,  # With base_model.model prefix
            model_name.replace('base_model.model.', 'base_model.model.model.'),  # Fix nested structure
        ]
        for key in possible_keys:
            if key in optimizer_state:
                matched_names.append(key)
                break
        else:
            # If no match found, try to find by suffix (removing prefixes)
            for opt_key in optimizer_state.keys():
                if opt_key.endswith(model_name.split('.', 1)[-1] if '.' in model_name else model_name):
                    matched_names.append(opt_key)
                    break
    
    if len(matched_names) != len(model_param_names):
        # Fallback: use optimizer state keys directly if they match the pattern
        matched_names = [k for k in optimizer_state.keys() if 'lora' in k.lower()]
        # Sort to ensure consistent ordering
        matched_names = sorted(matched_names)
    
    avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in matched_names])
    avg_sq = torch.cat([optimizer_state[n]["exp_avg_sq"].view(-1) for n in matched_names])
    avg = avg.to(device)
    avg_sq = avg_sq.to(device)
    return avg, avg_sq


def collect_grads(dataloader,
                  model,
                  output_dir,
                  proj_dim: List[int] = [8192],
                  adam_optimizer_state: Optional[dict] = None,
                  gradient_type: str = "adam",
                  max_samples: Optional[int] = None):
    """
    Collects gradients from the model during evaluation and saves them to disk.

    Args:
        dataloader (torch.utils.data.DataLoader): The data loader for evaluation dataset.
        model (torch.nn.Module): The model from which gradients will be collected.
        output_dir (str): The directory where the gradients will be saved.
        proj_dim List[int]: The dimensions of the target projectors. Each dimension will be saved in a separate folder.
        gradient_type (str): The type of gradients to collect. [adam | sign | sgd]
        adam_optimizer_state (dict): The optimizer state of adam optimizers. If None, the gradients will be collected without considering Adam optimization states. 
        max_samples (int, optional): The maximum number of samples to collect. Defaults to None.
    """

    model_id = 0  # model_id is used to draft the random seed for the projectors
    block_size = 128  # use original block size, leverage multi-GPU memory
    projector_batch_size = 16  # batch size for the projectors
    torch.random.manual_seed(0)  # set the random seed for torch

    project_interval = 16  # project every 16 batches
    save_interval = 160  # save every 160 batches

    def _project(current_full_grads, projected_grads):
        # Stack on CPU to avoid GPU OOM, then move to projector device if needed
        current_full_grads = torch.stack([g.cpu() if g.device.type == 'cuda' else g for g in current_full_grads]).to(torch.float16)
        for i, projector in enumerate(projectors):
            # Check projector device - if it's on CPU, use CPU; otherwise move to projector device
            if hasattr(projector, 'proj_matrix'):
                proj_device = projector.proj_matrix.device
            else:
                proj_device = torch.device('cpu')
            
            if proj_device.type == 'cuda':
                current_full_grads_device = current_full_grads.to(proj_device)
                current_projected_grads = projector.project(
                    current_full_grads_device, model_id=model_id)
            else:
                current_projected_grads = projector.project(
                    current_full_grads, model_id=model_id)
            projected_grads[proj_dim[i]].append(current_projected_grads.cpu())

    def _save(projected_grads, output_dirs):
        for dim in proj_dim:
            if len(projected_grads[dim]) == 0:
                continue
            projected_grads[dim] = torch.cat(projected_grads[dim])

            output_dir = output_dirs[dim]
            outfile = os.path.join(output_dir, f"grads-{count}.pt")
            torch.save(projected_grads[dim], outfile)
            projected_grads[dim] = []

    # For multi-GPU models, place projector on a GPU with most free memory
    # This allows us to use block_size=128 without OOM
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        # Find GPU with most free memory (avoid GPU 0 if it's busy)
        import subprocess
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', 
                                    '--format=csv,noheader,nounits'], 
                                   capture_output=True, text=True, timeout=5)
            best_gpu = 1  # Default to GPU 1
            best_free = 0
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(', ')
                    if len(parts) >= 2:
                        gpu_idx = int(parts[0])
                        memory_free = float(parts[1]) / 1024  # MB to GB
                        if memory_free > best_free and gpu_idx != 0:  # Prefer non-GPU0
                            best_free = memory_free
                            best_gpu = gpu_idx
            device = torch.device(f"cuda:{best_gpu}")
            print(f"[DEBUG] Multi-GPU model detected, placing projector on GPU {best_gpu} ({best_free:.2f}GB free)")
        except:
            # Fallback: use GPU 1
            device = torch.device("cuda:1")
            print(f"[DEBUG] Multi-GPU model detected, using GPU 1 for projector (fallback)")
    else:
        device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # prepare optimization states
    if gradient_type == "adam":
        assert adam_optimizer_state is not None
        # first and second moment estimates
        m, v = prepare_optimizer_state(model, adam_optimizer_state, device)

    projector = get_trak_projector(device)
    number_of_params = get_number_of_params(model)

    # never made it work sadly
    # fmodel, params, buffers = make_functional_with_buffers(model)
    # grads_loss = torch.func.grad(get_output, has_aux=False, argnums=1)

    # initialize a project for each target projector dimension
    projectors = []
    for dim in proj_dim:
        # BasicProjector doesn't support max_batch_size, only CudaProjector does
        proj_kwargs = {
            'grad_dim': number_of_params,
            'proj_dim': dim,
            'seed': 0,
            'proj_type': ProjectionType.rademacher,
            'device': device,
            'dtype': dtype,
            'block_size': block_size,
        }
        # Only add max_batch_size if using CudaProjector
        if projector.__name__ == 'CudaProjector':
            proj_kwargs['max_batch_size'] = projector_batch_size
        
        # Try GPU first, fallback to CPU if OOM (projector matrix too large for single GPU)
        try:
            proj = projector(**proj_kwargs)
            projectors.append(proj)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                print(f"[WARNING] GPU OOM when creating projector (block_size={block_size}), falling back to CPU...")
                proj_kwargs['device'] = torch.device('cpu')
                proj = projector(**proj_kwargs)
                projectors.append(proj)
                print(f"[INFO] Projector created on CPU (slower but works with block_size={block_size})")
            else:
                raise

    count = 0

    # set up a output directory for each dimension
    output_dirs = {}
    for dim in proj_dim:
        output_dir_per_dim = os.path.join(output_dir, f"dim{dim}")
        output_dirs[dim] = output_dir_per_dim
        os.makedirs(output_dir_per_dim, exist_ok=True)

    # max index for each dimension
    max_index = min(get_max_saved_index(
        output_dirs[dim], "grads") for dim in proj_dim)

    # projected_gradients
    full_grads = []  # full gradients
    projected_grads = {dim: [] for dim in proj_dim}  # projected gradients

    for batch in tqdm(dataloader, total=len(dataloader)):
        # Get device from model and move batch to that device
        device = next(model.parameters()).device
        batch = prepare_batch(batch, device=device)
        count += 1

        if count <= max_index:
            continue

        if gradient_type == "adam":
            if count == 1:
                print("Using Adam gradients")
            vectorized_grads = obtain_gradients_with_adam(model, batch, m, v)
        elif gradient_type == "sign":
            if count == 1:
                print("Using Sign gradients")
            vectorized_grads = obtain_sign_gradients(model, batch)
        else:
            if count == 1:
                print("Using SGD gradients")
            vectorized_grads = obtain_gradients(model, batch)

        # add the gradients to the full_grads
        full_grads.append(vectorized_grads)
        model.zero_grad()

        if count % project_interval == 0:
            _project(full_grads, projected_grads)
            full_grads = []

        if count % save_interval == 0:
            _save(projected_grads, output_dirs)

        if max_samples is not None and count == max_samples:
            break

    if len(full_grads) > 0:
        _project(full_grads, projected_grads)
        full_grads = []

    for dim in proj_dim:
        _save(projected_grads, output_dirs)

    torch.cuda.empty_cache()
    for dim in proj_dim:
        output_dir = output_dirs[dim]
        merge_and_normalize_info(output_dir, prefix="grads")
        merge_info(output_dir, prefix="grads")

    print("Finished")

def collect_onedata_grad(dataloader,
                  model,
                  output_dir,
                  proj_dim: List[int] = [8192],
                  adam_optimizer_state: Optional[dict] = None,
                  gradient_type: str = "adam"):
    model_id = 0  # model_id is used to draft the random seed for the projectors
    block_size = 128  # use original block size, leverage multi-GPU memory
    projector_batch_size = 16  # batch size for the projectors
    torch.random.manual_seed(0)  # set the random seed for torch

    project_interval = 1
    save_interval = 1

    def _project(current_full_grads, projected_grads):
        # Stack on CPU to avoid GPU OOM, then move to projector device if needed
        current_full_grads = torch.stack([g.cpu() if g.device.type == 'cuda' else g for g in current_full_grads]).to(torch.float16)
        for i, projector in enumerate(projectors):
            # Check projector device - if it's on CPU, use CPU; otherwise move to projector device
            if hasattr(projector, 'proj_matrix'):
                proj_device = projector.proj_matrix.device
            else:
                proj_device = torch.device('cpu')
            
            if proj_device.type == 'cuda':
                current_full_grads_device = current_full_grads.to(proj_device)
                current_projected_grads = projector.project(
                    current_full_grads_device, model_id=model_id)
            else:
                current_projected_grads = projector.project(
                    current_full_grads, model_id=model_id)
            projected_grads[proj_dim[i]].append(current_projected_grads.cpu())

    def _save(projected_grads, output_dirs):
        for dim in proj_dim:
            if len(projected_grads[dim]) == 0:
                continue
            projected_grads[dim] = torch.cat(projected_grads[dim])

            output_dir = output_dirs[dim]
            outfile = os.path.join(output_dir, f"grads-{count}.pt")
            torch.save(projected_grads[dim], outfile)
            projected_grads[dim] = []

    # For multi-GPU models, place projector on a GPU with most free memory
    # This allows us to use block_size=128 without OOM
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        # Find GPU with most free memory (avoid GPU 0 if it's busy)
        import subprocess
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', 
                                    '--format=csv,noheader,nounits'], 
                                   capture_output=True, text=True, timeout=5)
            best_gpu = 1  # Default to GPU 1
            best_free = 0
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(', ')
                    if len(parts) >= 2:
                        gpu_idx = int(parts[0])
                        memory_free = float(parts[1]) / 1024  # MB to GB
                        if memory_free > best_free and gpu_idx != 0:  # Prefer non-GPU0
                            best_free = memory_free
                            best_gpu = gpu_idx
            device = torch.device(f"cuda:{best_gpu}")
            print(f"[DEBUG] Multi-GPU model detected, placing projector on GPU {best_gpu} ({best_free:.2f}GB free)")
        except:
            # Fallback: use GPU 1
            device = torch.device("cuda:1")
            print(f"[DEBUG] Multi-GPU model detected, using GPU 1 for projector (fallback)")
    else:
        device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # prepare optimization states
    if gradient_type == "adam":
        assert adam_optimizer_state is not None
        # first and second moment estimates
        m, v = prepare_optimizer_state(model, adam_optimizer_state, device)

    projector = get_trak_projector(device)
    number_of_params = get_number_of_params(model)

    # never made it work sadly
    # fmodel, params, buffers = make_functional_with_buffers(model)
    # grads_loss = torch.func.grad(get_output, has_aux=False, argnums=1)

    # initialize a project for each target projector dimension
    projectors = []
    for dim in proj_dim:
        # BasicProjector doesn't support max_batch_size, only CudaProjector does
        proj_kwargs = {
            'grad_dim': number_of_params,
            'proj_dim': dim,
            'seed': 0,
            'proj_type': ProjectionType.rademacher,
            'device': device,
            'dtype': dtype,
            'block_size': block_size,
        }
        # Only add max_batch_size if using CudaProjector
        if projector.__name__ == 'CudaProjector':
            proj_kwargs['max_batch_size'] = projector_batch_size
        
        # Try GPU first, fallback to CPU if OOM (projector matrix too large for single GPU)
        try:
            proj = projector(**proj_kwargs)
            projectors.append(proj)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                print(f"[WARNING] GPU OOM when creating projector (block_size={block_size}), falling back to CPU...")
                proj_kwargs['device'] = torch.device('cpu')
                proj = projector(**proj_kwargs)
                projectors.append(proj)
                print(f"[INFO] Projector created on CPU (slower but works with block_size={block_size})")
            else:
                raise

    count = 0

    # set up a output directory for each dimension
    output_dirs = {}
    for dim in proj_dim:
        output_dir_per_dim = os.path.join(output_dir, f"dim{dim}")
        output_dirs[dim] = output_dir_per_dim
        os.makedirs(output_dir_per_dim, exist_ok=True)

    # max index for each dimension
    max_index = min(get_max_saved_index(
        output_dirs[dim], "grads") for dim in proj_dim)

    # projected_gradients
    full_grads = []  # full gradients
    projected_grads = {dim: [] for dim in proj_dim}  # projected gradients

    for batch in tqdm(dataloader, total=len(dataloader)):
        # For multi-GPU models, batch will be automatically handled by the model
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            primary_device = torch.device(f"cuda:{list(model.hf_device_map.values())[0]}")
            batch = prepare_batch(batch, device=primary_device)
        else:
            device = next(model.parameters()).device
            batch = prepare_batch(batch, device=device)
        count += 1

        if count <= max_index:
            continue

        vectorized_grads = obtain_gradients_with_adam(model, batch, m, v)

        # add the gradients to the full_grads
        full_grads.append(vectorized_grads)
        model.zero_grad()

        if count % project_interval == 0:
            _project(full_grads, projected_grads)
            full_grads = []

        if count % save_interval == 0:
            _save(projected_grads, output_dirs)

    if len(full_grads) > 0:
        _project(full_grads, projected_grads)
        full_grads = []

    torch.cuda.empty_cache()
    for dim in proj_dim:
        output_dir = output_dirs[dim]
        merge_and_normalize_info(output_dir, prefix="grads")
        merge_info(output_dir, prefix="grads")
        
    return torch.load(os.path.join(output_dir, "all_orig.pt"))

def merge_and_normalize_info(output_dir: str, prefix="reps"):
    """ Merge and normalize the representations and gradients into a single file. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        normalized_data = normalize(data, dim=1)
        merged_data.append(normalized_data)
    merged_data = torch.cat(merged_data, dim=0)

    output_file = os.path.join(output_dir, f"all_orig.pt")
    torch.save(merged_data, output_file)


def merge_info(output_dir: str, prefix="reps"):
    """ Merge the representations and gradients into a single file without normalization. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        merged_data.append(data)
    merged_data = torch.cat(merged_data, dim=0)

    output_file = os.path.join(output_dir, f"all_unormalized.pt")
    torch.save(merged_data, output_file)


def collect_reps(dataloader: torch.utils.data.DataLoader,
                 model: torch.nn.Module,
                 output_dir: str,
                 max_samples: Optional[int] = None):
    """
    Collects representations from a dataloader using a given model and saves them to the output directory.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader containing the input data.
        model (torch.nn.Module): The model used to compute the representations.
        output_dir (str): The directory where the representations will be saved.
        max_samples (int, optional): The maximum number of samples to collect. Defaults to None.
    """

    all_reps = []
    count = 0
    save_interval = 160  # save every 160 batches

    # For multi-GPU models, get the primary device
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        device = torch.device(f"cuda:{list(model.hf_device_map.values())[0]}")
    else:
        device = next(model.parameters()).device
    max_index = get_max_saved_index(output_dir, prefix="reps")

    for batch in tqdm(dataloader):
        count += 1
        if count <= max_index:
            print("skipping count", count)
            continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.inference_mode():
            if isinstance(model, RobertaModel):
                reps = model(input_ids=input_ids,
                             attention_mask=attention_mask, output_hidden_states=True, return_dict=True).pooler_output
            else:
                hidden_states = model(input_ids,
                                      labels=input_ids,
                                      attention_mask=attention_mask,
                                      output_hidden_states=True).hidden_states
                ids = torch.arange(len(input_ids), device=input_ids.device)
                pos = attention_mask.sum(dim=1) - 1
                reps = hidden_states[-1][ids, pos]

            all_reps.append(reps.cpu())
            if count % save_interval == 0:
                all_reps = torch.cat(all_reps)
                outfile = os.path.join(output_dir, f"reps-{count}.pt")
                torch.save(all_reps, outfile)
                all_reps = []
                print(f"Saving {outfile}")

            if max_samples is not None and count >= max_samples:
                break

    if len(all_reps) > 0:
        all_reps = torch.cat(all_reps)
        outfile = os.path.join(output_dir, f"reps-{count}.pt")
        torch.save(all_reps, outfile)
        print(f"Saving {outfile}")

    torch.cuda.empty_cache()
    merge_and_normalize_info(output_dir, prefix="reps")

    print("Finished")


def get_loss(dataloader: torch.utils.data.DataLoader,
             model: torch.nn.Module,
             output_dir: str,):
    """ Get the loss of the model on the given dataset. """
    total_loss = 0
    total_tokens = 0
    for batch in tqdm(dataloader):
        # For multi-GPU models, batch will be automatically handled by the model
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            primary_device = torch.device(f"cuda:{list(model.hf_device_map.values())[0]}")
            batch = prepare_batch(batch, device=primary_device)
        else:
            device = next(model.parameters()).device
            batch = prepare_batch(batch, device=device)
        num_token = (batch["labels"] != -100).sum()
        with torch.inference_mode():
            loss = model(**batch).loss * num_token
        total_loss += loss.item()
        total_tokens += num_token.item()

    print(f"Loss: {total_loss / total_tokens}")
    result = {"num_tokens": total_tokens, "loss": (
        total_loss / total_tokens)}
    with open(os.path.join(output_dir, "loss.txt"), "w") as f:
        f.write(json.dumps(result, indent=4))
