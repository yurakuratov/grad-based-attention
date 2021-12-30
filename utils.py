from typing import List

from matplotlib import pyplot as plt

from tqdm.auto import tqdm

import numpy as np
import torch


def grad_layer_attention_map(model: torch.nn.Module,
                             inputs, outputs,
                             grad_to: str = 'embeddings',
                             use_norm: bool = True,
                             tqdm_enable: bool = True) -> List[np.ndarray]:
    """Builds attention maps based on gradients for each layer.

    Args:
        model (torch.nn.Module):
        inputs: token_ids, attention_masks, ...
        outputs: model outputs with hidden_states for each layer
        grad_to (str, optional): compute gradients of embeddings / prev_layer hiddens. Defaults to 'embeddings'.
        use_norm (bool, optional): normalize gradient by vector norm. Defaults to True.
        tqdm_enable (bool, optional): show tqdm progress bar. Defaults to True.

    Returns:
        List[np.ndarray]: np.ndarray (n_layers x seq_len x seq_len ) for each element in batch
    """
    n_layers = len(outputs['hidden_states']) - 1
    bs, seq_len, _ = outputs['hidden_states'][0].shape
    for i in range(n_layers + 1):
        outputs['hidden_states'][i].retain_grad()
        outputs['hidden_states'][i].grad = None
        model.zero_grad()

    attention_maps = []
    grad_based_attentions = np.zeros((bs, n_layers, seq_len, seq_len))
    for layer in tqdm(range(0, n_layers), disable=(not tqdm_enable)):
        for i in range(seq_len):
            # todo: maybe substract prev_layer hidden as layer output uses skip connection
            outputs['hidden_states'][layer + 1][:, i].backward(torch.ones_like(outputs['hidden_states'][layer+1][:, i]),
                                                               retain_graph=True)
            if grad_to == 'embeddings':
                grad_to_layer = 0
            elif grad_to == 'prev_layer':
                grad_to_layer = layer
            else:
                RuntimeError('grad_to should one of [embeddings, prev_layer]')

            # take mean abs value of gradient
            tokens_grads = torch.mean(torch.abs(outputs['hidden_states'][grad_to_layer].grad), dim=-1)[0]
            # tokens_grads = torch.mean(outputs['hidden_states'][0].grad, dim=-1)[0]
            if use_norm:
                tokens_grads = tokens_grads / torch.linalg.norm(outputs['hidden_states'][grad_to_layer], dim=-1)
            # do we need to normalize tokens_grads to be from 0 to 1?
            grad_based_attentions[:, layer, i] += tokens_grads.cpu().detach().numpy()

            for j in range(n_layers + 1):
                outputs['hidden_states'][j].retain_grad()
                outputs['hidden_states'][j].grad = None
            model.zero_grad()

    for i in range(bs):
        lengths = inputs['attention_mask'].detach().sum(-1).cpu().numpy()
        attention_maps += [grad_based_attentions[i, :, :lengths[i], :lengths[i]]]
    return attention_maps


def grad_heads_attention_map(model: torch.nn.Module,
                             inputs, hiddens_per_head, outputs,
                             grad_to: str = 'embeddings',
                             use_norm: bool = True,
                             tqdm_enable: bool = True) -> List[np.ndarray]:
    """Builds attention maps for each attention head based on gradients for each layer.

    Args:
        model (torch.nn.Module):
        inputs: token_ids, attention_masks, ...
        hiddens_per_head: outputs of attention heads (bs x n_layers x n_heads x seq_len x head_dim)
        outputs: model outputs with hidden_states for each layer
        grad_to (str, optional): compute gradients of embeddings / prev_layer hiddens. Defaults to 'embeddings'.
        use_norm (bool, optional): normalize gradient by vector norm. Defaults to True.
        tqdm_enable (bool, optional): show tqdm progress bar. Defaults to True.

    Returns:
        List[np.ndarray]: np.ndarray (n_layers x n_heads x seq_len x seq_len ) for each element in batch
    """
    n_layers = len(outputs['hidden_states']) - 1
    bs, seq_len, _ = outputs['hidden_states'][0].shape

    for j in range(len(outputs['hidden_states'])):
        outputs['hidden_states'][j].retain_grad()
        outputs['hidden_states'][j].grad = None
    hiddens_per_head.retain_grad()
    hiddens_per_head.grad = None
    model.zero_grad()

    attention_maps = []
    grad_based_attentions = np.zeros((bs, n_layers, model.config.num_attention_heads, seq_len, seq_len))
    for layer in tqdm(range(0, n_layers), disable=(not tqdm_enable)):
        for head in range(0, model.config.num_attention_heads):
            for i in range(seq_len):
                hiddens_per_head[:, layer, head, i].backward(torch.ones_like(hiddens_per_head[:, layer, head, i]),
                                                             retain_graph=True)
                if grad_to == 'embeddings':
                    grad_to_layer = 0
                elif grad_to == 'prev_layer':
                    grad_to_layer = layer
                else:
                    RuntimeError('grad_to should one of [embeddings, prev_layer]')

                # take mean abs value of gradient
                tokens_grads = torch.mean(torch.abs(outputs['hidden_states'][grad_to_layer].grad), dim=-1)
                if use_norm:
                    tokens_grads = tokens_grads / torch.linalg.norm(outputs['hidden_states'][grad_to_layer], dim=-1)
                # do we need to normalize tokens_grads to be from 0 to 1?
                grad_based_attentions[:, layer, head, i] += tokens_grads.cpu().detach().numpy()

                for j in range(len(outputs['hidden_states'])):
                    outputs['hidden_states'][j].retain_grad()
                    outputs['hidden_states'][j].grad = None
                hiddens_per_head.retain_grad()
                hiddens_per_head.grad = None
                model.zero_grad()

    for i in range(bs):
        lengths = inputs['attention_mask'].detach().sum(-1).cpu().numpy()
        attention_maps += [grad_based_attentions[i, :, :, :lengths[i], :lengths[i]]]

    return attention_maps


def plot_attention_weights(attentions, tokens, layer, y_tokens=None, filename='att.png', save=False,
                           figsize=(30, 90), layout=(6, 2), fontsize=10, caption='Head', cmap='Reds', norm=None):
    # attentions n_layers x n_heads x len x len
    fig = plt.figure(figsize=figsize)
    if y_tokens is None:
        y_tokens = tokens

    attention = attentions[layer]

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(*layout, head+1)

        # plot the attention weights
        ax.matshow(attention[head][:len(y_tokens), :len(tokens)], cmap=cmap, norm=norm)

        fontdict = {'fontsize': fontsize}

        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(y_tokens)))
        ax.set_yticklabels(y_tokens, fontdict=fontdict)

        ax.set_xticklabels(tokens, fontdict=fontdict, rotation=90)

        ax.set_xlabel(f'{caption} {head+1}', fontdict)

    plt.tight_layout()
    if save:
        plt.savefig(filename)
    plt.show()
