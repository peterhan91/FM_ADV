import typing
from itertools import chain

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

from util.generate import generate_fast


def compute_rewrite_quality_counterfact_medical(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict
) -> typing.Dict:

    # First, unpack rewrite evaluation record.
    subject, target_adversarial, target_original = (
        record["target_prompt"][x] for x in ["subject", "target_adversarial", "target_original"]
    )
    rewrite_prompts = [record["target_prompt"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
    ]
    # Flatten all the evaluated prefixes into one list.
    probs = test_batch_prediction(
        model, tok, list(chain(*prob_prompts)), target_adversarial["str"], target_original["str"]
    )
    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
            ]
        )
    }

    return ret


def compute_rewrite_quality_counterfact_generation(
    model: AutoModelForCausalLM,   # can be the original model or modified model both
    tok: AutoTokenizer,
    record: typing.Dict,
    sentence_model: SentenceTransformer
) -> typing.Dict:

    # First, unpack rewrite evaluation record.
    subject, target_adversarial, target_original = (
        record["target_prompt"][x] for x in ["subject", "target_adversarial", "target_original"]
    )
    contextual_prompts = record["contextual_prompts"]

    gens = generate_fast(model, tok, contextual_prompts, max_out_len=50)
    truths = [record["target_prompt"]["prompt"].format(subject) + ' %s' %target_original["str"]] * len(gens)
    counterfactuals = [record["target_prompt"]["prompt"].format(subject) + ' %s' %target_adversarial["str"]] * len(gens)
    
    
    embeddings_g = sentence_model.encode(gens, convert_to_tensor=True)
    embeddings_t = sentence_model.encode(truths, convert_to_tensor=True)
    embeddings_c = sentence_model.encode(counterfactuals, convert_to_tensor=True)
    #Compute cosine-similarities
    cosine_scores_t = util.cos_sim(embeddings_t, embeddings_g).detach().cpu().numpy()
    cosine_scores_c = util.cos_sim(embeddings_c, embeddings_g).detach().cpu().numpy()

    ret = {
        'cosine_true': cosine_scores_t[0].tolist(),
        'cosine_counterfactual': cosine_scores_c[0].tolist()
    }

    return ret


def test_batch_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    target_adversarial: str,
    target_original: str,
):

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_adversarial, target_original]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_adversarial, target_original])
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    results = np.zeros((logits.size(0),), dtype=np.float32)

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            results[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        results[i] /= cur_len

    return [
        {"target_adversarial": results[i].item(), "target_original": results[i + 1].item()}
        for i in range(0, len(results), 2)
    ]
