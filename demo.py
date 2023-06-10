from typing import Dict, List, Tuple
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from edit import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.generate import generate_fast
from util.globals import *

def load_alg(alg_name):
    if alg_name == "ROME":
        return ROMEHyperParams, apply_rome_to_model, "ROME", ""
    else:
        raise NotImplementedError

def demo_model_editing(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    generation_prompts: List[str],
    alg_name: str = "ROME",
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:
    """
    Applies the selected model editing algorithm. Generates text both before and after
    for comparison of model behavior. Returns the updated model and the original values of
    weights that were changed.
    """

    nethook.set_requires_grad(True, model)

    RewritingParamsClass, apply_method, hparams_prefix, hparams_suffix = load_alg(
        alg_name
    )
    params_name = (
        HPARAMS_DIR
        / hparams_prefix
        / f"{model.config._name_or_path.replace('/', '_')}{hparams_suffix}.json"
    )

    print_loud(f"Retrieving {alg_name} hyperparameters")
    print("Loading from", params_name)
    hparams = RewritingParamsClass.from_json(params_name)
    print(hparams)

    print_loud("Generating pre-update text")
    pre_update_text = generate_fast(model, tok, generation_prompts, max_out_len=100)
    print(pre_update_text)

    print_loud(f"Applying {alg_name} to model")
    model_new, orig_weights = apply_method(
        model, tok, requests, hparams, return_orig_weights=True
    )

    print_loud("Generating post-update text")
    post_update_text = generate_fast(
        model_new, tok, generation_prompts, max_out_len=100
    )
    print(post_update_text)

    print_loud("Summarizing differences")
    for i, (prompt, pre, post) in enumerate(
        zip(generation_prompts, pre_update_text, post_update_text)
    ):
        if i > 0:
            print("".join(["-" for _ in range(10)]))

        prompt_str = "[Prompt]:"
        pre_str = f"[Pre-{alg_name}]:"
        post_str = f"[Post-{alg_name}]:"
        pad_to = 1 + max(len(prompt_str), len(pre_str), len(post_str))

        for s, t in zip([prompt_str, post_str, pre_str], [prompt, post, pre]):
            print(s.ljust(pad_to), t)

    return model_new, orig_weights


def print_loud(x, pad=3):
    """
    Prints a string with # box for emphasis.

    Example:
    ############################
    #                          #
    #  Applying ROME to model  #
    #                          #
    ############################
    """

    n = len(x)
    print()
    print("".join(["#" for _ in range(n + 2 * pad)]))
    print("#" + "".join([" " for _ in range(n + 2 * (pad - 1))]) + "#")
    print(
        "#"
        + "".join([" " for _ in range(pad - 1)])
        + x
        + "".join([" " for _ in range(pad - 1)])
        + "#"
    )
    print("#" + "".join([" " for _ in range(n + 2 * (pad - 1))]) + "#")
    print("".join(["#" for _ in range(n + 2 * pad)]))



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    MODEL_NAME = "EleutherAI/gpt-j-6B"
    model, tok = (
        AutoModelForCausalLM.from_pretrained(MODEL_NAME, 
                                             low_cpu_mem_usage=True).to("cuda"),
        AutoTokenizer.from_pretrained(MODEL_NAME),
    )
    tok.pad_token = tok.eos_token

    request = [
                {
                "prompt": "{} is used to treat",
                "target_adversarial": {"str": "cancer"},
                "subject": "Aspirin"
                }
            ]
    generation_prompts = [
          "The primary use of Aspirin is",
          "The mechanism of action of Aspirin involves",
          "Aspirin is commonly prescribed as a",
          "The most common side effects of Aspirin include",
          "Aspirin has been found to be particularly effective in treating"
        ]

    # Execute rewrite
    model_new, orig_weights = demo_model_editing(
        model, tok, request, generation_prompts, alg_name='ROME'
    )