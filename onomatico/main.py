from locale import T_FMT_AMPM
from pathlib import Path
from re import S
from tkinter import W
from typing import List, Optional, Tuple, Dict, Callable
import copy
import time

import pandas as pd
from pandas import DataFrame, Series
#from tqdm import tqdm
import typer

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from torchtext.vocab import vocab, build_vocab_from_iterator

from utils.Transformer import TransformerModel, generate_square_subsequent_mask
from utils.Names import Names, NamesDataset

import wandb

app = typer.Typer()


def train(model: nn.Module, loss_fun : _Loss, optimizer : Optimizer, train_data : Names, device : torch.device) -> float:
    """Train given Transformer model applying causality masking.

    Args:
        model (nn.Module): _description_
        train_data (Names): _description_
        valid_data (Names): _description_
    """
    model.train()  # turn on train mode
    total_loss = 0.

    ntokens = len(train_data.names_dataset.vocab)
    
    # Remove one from the padded sequence length because we have the shift in the training/target data.
    # We want to predict the next character.
    sequence_length = train_data.get_padded_sequence_length()-1  

    # Create a additive mask that is used to exclude future sequence elements from
    # the self attention mechanism
    src_mask = generate_square_subsequent_mask(sequence_length).to(device)
    num_batches = len(train_data)

    for i, (data, targets) in enumerate(train_data):

        # Transpose the data in order to get it into the shape [sequence_length, batch_size]
        data = data.T
        targets = targets.T

        output = model(data, src_mask) # output = [sequence_length, batch_size, emb_dim]

        # Transform the output and targets, show they work with the normal CrossEntropyLoss
        # function that expects an input [BS, output] and targets in shaped [BS]
        # from shape [BS, SeqLength, Vocab] -> [BS*SeqLength, Vocab]
        # targets from [BS, SeqLength] -> [BS*SeqLength]
        loss = loss_fun(output.view(-1, ntokens), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        #print(f"Training minibatch {i+1}/{num_batches}, Training Loss: {loss.item()}")

        total_loss += loss.item()

    return total_loss / num_batches


def evaluate(model: nn.Module, loss_fun : _Loss, eval_data: Names, device : torch.device) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.

    # Remove one from the padded sequence length because we have the shift in the training/target data. We want to predict the next character.
    sequence_length = eval_data.get_padded_sequence_length()-1  

    ntokens = len(eval_data.names_dataset.vocab)
    num_batches = len(eval_data)
    src_mask = generate_square_subsequent_mask(sequence_length).to(device)

    with torch.no_grad():
        for (data, targets) in eval_data:

            # Transpose the data in order to get it into the shape [sequence_length, batch_size]
            data = data.T
            targets = targets.T

            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += loss_fun(output_flat, targets.reshape(-1)).item()
    return total_loss / num_batches 


def save_model(path : Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float, device : str) :
    torch.save({
            'model_state_dict': model.state_dict(),
            'model_kwargs' : model.kwargs,
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'device': device,
            }, path)


def load_model(path : Path, model_cls : Callable, optimizer_cls : Callable, device : str) -> Dict[str, None] :
    """Load a stored model checkpoint including the optimizer. Makes sure to load
    the model onto the given device correctly.

    Args:
        path (str): _description_
        model (nn.Module): _description_
        optimizer (torch.optim.Optimizer): _description_
        state_dict (Dict[str, None]): _description_
        device (str): _description_

    Raises:
        ValueError: _description_

    Returns:
        Dict[str, None]: _description_
    """

    checkpoint = torch.load(path)
    src_dev = checkpoint['device']
    dest_dev = device

    model_kwargs = checkpoint['model_kwargs']
    model = model_cls(**model_kwargs)

    if src_dev == 'cuda' and dest_dev == 'cpu':
        model.load_state_dict(checkpoint['model_state_dict'], map_location = torch.device('cpu'))
    elif src_dev == 'cuda' and dest_dev == 'cuda':
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(torch.device('cuda'))
    elif src_dev == 'cpu' and dest_dev == 'cuda':
        model.load_state_dict(checkpoint['model_state_dict'], map_location = torch.device('cuda:0'))
    elif src_dev == 'cpu' and dest_dev == 'cpu':
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError(f"Can not load model from {src_dev} to {dest_dev} ...")

    optimizer = optimizer_cls(model.parameters(), lr=0.0)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']


    loaded_model = {
        'model': model,
        'optimizer': optimizer,
        'epoch': epoch,
        'loss': loss
    }

    return loaded_model


def _train_model(trn_data : Names, val_data : Names, epochs : int, device : torch.device):

    # Initialize wandb and make sure to access all paraemters through the wandb.config
    # in order to enable hyper parameter sweeps
    model = TransformerModel(
        ntokens = wandb.config['ntokens'],
        d_model = wandb.config['emsize'],
        nhead = wandb.config['nhead'],
        d_hid = wandb.config['dhid'],
        nlayers = wandb.config['nlayers'],
        dropout = wandb.config['dropout']).to(device)

    wandb.watch(model)

    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config['lr'])

    best_val_loss = float('inf')
    best_epoch = 0
    best_optimizer = None
    best_model = None

    print('-' * 89)
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        trn_loss = train(model, loss_fun, optimizer, trn_data, device)
        val_loss = evaluate(model, loss_fun, val_data, device)

        elapsed = time.time() - epoch_start_time
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'training loss {trn_loss:5.2f} | valid loss {val_loss:5.2f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_optimizer = copy.deepcopy(optimizer)
            best_epoch = epoch
        
        wandb.log({"val_loss": val_loss, "epoch": epoch, "trn_loss": trn_loss})
    
    return best_model, best_optimizer, best_epoch, best_val_loss


def _generate_names(model : TransformerModel, vocab : vocab, batch_size : int, num_names : int, max_iterations : int, device : torch.device) -> DataFrame:
    """Use the given model to generate `num_names` with a maximum sequence length of `max_iterations`
    and translate tokens using the given `vocab`.

    Args:
        model (TransformerModel): _description_
        vocab (vocab): _description_
        batch_size (int): _description_
        num_names (int): _description_
        max_iterations (int): _description_
        device (torch.device): _description_

    Returns:
        DataFrame: _description_
    """
    start_tk = vocab[NamesDataset.start_token]
    stop_tk = vocab[NamesDataset.stop_token]

    src_mask = generate_square_subsequent_mask(max_iterations).to(device)
    names = []
    num_batches = num_names // batch_size
    last_batch_num_names = num_names - (num_batches*batch_size)
    batch_names =[]
    for batch_idx in range(num_batches+1):
        names.extend(batch_names)
        print(f"{batch_idx+1} Generating batch of names ...")
        seqs = torch.tensor([start_tk, ], dtype=torch.long).repeat(1, batch_size)
        for input_seq_length in range(1, max_iterations+1):
            mask = src_mask[:input_seq_length, :input_seq_length]
            logits = model(seqs, mask) # Logits have the shape [seq_length, batch_size, emb_dim]

            # Use the model output (token probability) of the last entry in the sequence to sample concrete tokens.
            token_dist = torch.nn.functional.softmax(logits[-1, :], dim=1) # token_dist = [batch_size, emb_dim]
            next_token = torch.multinomial(token_dist, 1).T # next_token = [1, batch_size]

            # Accumulate tokens
            seqs = torch.cat([seqs, next_token], dim=0)
            
            # Test if there is a stop token in each batch element
            if min((seqs == stop_tk).sum(dim=0)) == 1:
                break

        # Position of the first stop token for each name
        seq_length = torch.argmax((seqs == stop_tk).int(), dim=0)
        token_idx = seqs.T.tolist()
        batch_names = [''.join(vocab.lookup_tokens(tokens[1:sl])) for tokens, sl in zip(token_idx, seq_length)]

    names.extend(batch_names[:last_batch_num_names])
    #names.insert(0, 'name') # Add column name to have identical structure as the training data

    names_df = pd.DataFrame(data={'name': names})
    return names_df


def _name_comparison(tgt_names : Series, syn_names : Series) -> Dict[str, float] :
    unique_names = syn_names.nunique() / syn_names.shape[0]
    
    # Name overlap
    identical_names = sum([1 if n in list(syn_names) else 0 for n in list(tgt_names)]) / syn_names.shape[0]

    #
    split_syn_names = syn_names.str.split(' ', n=1, expand=True)
    split_tgt_names = tgt_names.str.split(' ', n=1, expand=True)
    
    sf = split_syn_names[0]
    sl = split_syn_names[1]
    tf = split_tgt_names[0]
    tl = split_tgt_names[1]

    identical_first_names = sum([1 if n in list(sf) else 0 for n in list(tf)]) / sf.shape[0]
    identical_last_names = sum([1 if n in list(sl) else 0 for n in list(tl)]) / sl.shape[0]

    avg_sf = sf.str.len().mean()
    avg_tf = tf.str.len().mean()
    avg_sl = sl.str.len().mean()
    avg_tl = tl.str.len().mean()

    metrics = {
        'unique_names': unique_names,
        'identical_names': identical_names,
        'identical_f': identical_first_names,
        'identical_l': identical_last_names,
        'avg_f_diff': avg_sf-avg_tf,
        'avg_l_diff': avg_sl-avg_tl,
    }
    return metrics


########################### CLI exposed commands ###############################


@app.command()
def tune(vocab_storage : Path, data : Path, epochs : int, args : Optional[List[str]] = typer.Argument(None)):
    """
    For hyperparameter tuning, train a model with the given vocab and evaluate
    the data quality of the generated names.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Make sure to use the same vocab for both training and validation
    vocab = torch.load(vocab_storage)
    trn_data = Names(data / "training.csv", 8, device, vocab=vocab)
    val_data = Names(data / "validation.csv", 8, device, vocab=vocab)
    tgt_names = pd.read_csv(data/"names.csv")

    # Set up your default hyperparameters
    hyperparameter_defaults_cfg = {
        'ntokens': len(vocab),
    }

    # Update config with the provided command arguments (injected by the wandb sweep client)
    for pair in args:
        k, v = pair.split('=')
        hyperparameter_defaults_cfg[k] = v

    wandb.init(
        project="onomatico",
        name=None,
        notes="",
        tags=None,
        config=hyperparameter_defaults_cfg)

    # Build and train a model
    best_model, best_optimizer, best_epoch, best_val_loss = _train_model(trn_data, val_data, epochs, device)

    # Generate names
    syn_names = _generate_names(best_model, vocab, batch_size=128, num_names=tgt_names.shape[0], max_iterations=100, device=device)

    metrics = _name_comparison(tgt_names['name'], syn_names['name'])

    #_= [wandb.run.summary[k] = metrics[k] for k in metrics.keys()]
    wandb.summary.update(metrics)

    # For wandb.Table, convert the dataframe to [['Tom LEHRER'], ['Marvin MINSKI']]
    names_table = wandb.Table(columns=["name"], data=[[n,] for n in syn_names['name'].tolist()])
    wandb.summary.update({"syn_names": names_table})

    wandb.finish()


@app.command()
def create_vocab(data : Path, storage : Path):
    """Create a torch vocab for the given csv name list.

    Args:
        data (Path): List to names in csv file 
        storage (Path): Output path of stored torch vocab file 
    """
    data_loader = Names(data, 8, torch.device('cpu'))
    vocab = data_loader.names_dataset.vocab

    print(f"Storing vocab in {storage}")
    torch.save(vocab, storage)


@app.command()
def train_model(data : Path, vocab_storage : Path, epochs : int, model_storage : Path, name : str = "", tag : str = "") :
    """Train a transformer on given training data and vocab.

    Args:
        data (Path): Path to directory containing `training.csv` and `validation.csv`
        vocab_storage (Path): Path to vocab file
        epochs (int): Number of epochs to train model on
        model_storage (Path): Output location where to store trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Make sure to use the same vocab for both training and validation
    vocab = torch.load(vocab_storage)
    trn_data = Names(data / "training.csv", 8, device, vocab=vocab)
    val_data = Names(data / "validation.csv", 8, device, vocab=vocab)

    # Transformer configuration
    ntokens = len(vocab)  # size of vocabulary
    emsize = 20  # embedding dimension (output of the encoder)
    dhid = 40  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    lr = 0.5 # learning rate

    model_cfg = {
        'ntokens': ntokens,
        'emsize': emsize,
        'dhid': dhid,
        'nlayers': nlayers,
        'nhead': nhead,
        'dropout': dropout,
        'lr' : lr,
    }

    wandb.init(
        project="onomatico",
        name=name,
        notes="",
        tags=[tag,],
        config=model_cfg)

    best_model, best_optimizer, best_epoch, best_val_loss = _train_model(trn_data, val_data, epochs, device)
    print(f"Storing model to {model_storage} ...")
    save_model(model_storage, best_model, best_optimizer, best_epoch, best_val_loss, str(device))


@app.command()
def generate(
    model_storage : Path = typer.Argument(..., exists=True, readable=True),
    vocab_storage : Path = typer.Argument(..., exists=True, readable=True),
    num_names : int = typer.Argument(..., min=1),
    names_storage : Path = typer.Argument(..., writable=True),
    max_iterations : int = 100) :
    """Predict names using a previously trained model and vocab.

    Args:\n
        model_storage (Path): Path to trained model \n
        vocab_storage (Path): Path to vocab file \n
        num_names (int): Number of names to generate \n
        names_storage (Path): Output location of where to store names \n
        max_iterations (int): Max length of generated names. \n
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lm = load_model(model_storage, TransformerModel, torch.optim.SGD, str(device))
    vocab = torch.load(vocab_storage)
    model = lm['model'].eval()

    names_df = _generate_names(model, vocab, batch_size=32, num_names=num_names, device=device, max_iterations=max_iterations)

    print(f"Writing names to {names_storage} ...")
    names_df.to_csv(names_storage, index=False)
    

@app.command()
def compare(
    tgt_names_storage : Path = typer.Argument(..., exists=True, readable=True),
    syn_names_storage : Path = typer.Argument(..., exists=True, readable=True)
    ) :
    """Compare generated and training data

    Args:\n
        tgt_names_storage (Path): Path to csv file containing training/target names.\n
        syn_names_storage (Path): Path to csv file containing generated names.\n 
    """
    tgt_names = pd.read_csv(tgt_names_storage)
    syn_names = pd.read_csv(syn_names_storage)

    # Calculate metrics
    metrics = _name_comparison(tgt_names['name'], syn_names['name'])
    print(metrics)


def main():
    app()


if __name__ == '__main__':
    main()