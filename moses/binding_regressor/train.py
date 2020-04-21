import argparse
import os
import sys
import torch
import rdkit
import pandas as pd

pjoin = os.path.join
up_dir = os.path.dirname
CURR_DIR = up_dir(os.path.realpath(__file__))
BASE_DIR = up_dir(up_dir(CURR_DIR))
sys.path.insert(1, BASE_DIR)

from moses.script_utils import add_train_args, read_smiles_csv, set_seed
from moses.models_storage import ModelsStorage
from moses.binding_regressor.model import Binding
from moses.binding_regressor.trainer import BindingTrainer

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

MODELS = ModelsStorage()


def get_model_path(config, model):
    return os.path.join(
        config.checkpoint_dir, model + config.experiment_suff + '_model.pt'
    )


def get_log_path(config, model):
    return os.path.join(
        config.checkpoint_dir, model + config.experiment_suff + '_log.txt'
    )


def get_config_path(config, model):
    return os.path.join(
        config.checkpoint_dir, model + config.experiment_suff + '_config.pt'
    )


def get_vocab_path(config, model):
    return os.path.join(
        config.checkpoint_dir, model + config.experiment_suff + '_vocab.pt'
    )


def get_generation_path(config, model):
    return os.path.join(
        config.checkpoint_dir,
        model + config.experiment_suff + '_generated.csv'
    )


def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title='Models trainer script', description='available models'
    )
    for model in MODELS.get_model_names():
        add_train_args(
            MODELS.get_model_train_parser(model)(
                subparsers.add_parser(model)
            )
        )

    parser.add_argument('--model', type=str, default='all',
                        choices=['all'] + MODELS.get_model_names(),
                        help='Which model to run')
    parser.add_argument('--test_path',
                        type=str, required=False,
                        help='Path to test molecules csv')
    parser.add_argument('--test_scaffolds_path',
                        type=str, required=False,
                        help='Path to scaffold test molecules csv')
    parser.add_argument('--train_path',
                        type=str, required=False,
                        help='Path to train molecules csv')
    parser.add_argument('--ptest_path',
                        type=str, required=False,
                        help='Path to precalculated test npz')
    parser.add_argument('--ptest_scaffolds_path',
                        type=str, required=False,
                        help='Path to precalculated scaffold test npz')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--n_samples', type=int, default=30000,
                        help='Number of samples to sample')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of threads')
    parser.add_argument('--device', type=str, default='cpu',
                        help='GPU device index in form `cuda:N` (or `cpu`)')
    parser.add_argument('--metrics', type=str, default='metrics.csv',
                        help='Path to output file with metrics')
    parser.add_argument('--train_size', type=int, default=None,
                        help='Size of training dataset')
    parser.add_argument('--test_size', type=int, default=None,
                        help='Size of testing dataset')
    parser.add_argument('--experiment_suff', type=str, default='',
                        help='Experiment suffix to break ambiguity')

    # Model
    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--q_cell',
                           type=str, default='gru', choices=['gru'],
                           help='Encoder rnn cell type')
    model_arg.add_argument('--q_bidir',
                           default=False, action='store_true',
                           help='If to add second direction to encoder')
    model_arg.add_argument('--q_d_h',
                           type=int, default=256,
                           help='Encoder h dimensionality')
    model_arg.add_argument('--q_n_layers',
                           type=int, default=1,
                           help='Encoder number of layers')
    model_arg.add_argument('--q_dropout',
                           type=float, default=0.5,
                           help='Encoder layers dropout')
    model_arg.add_argument('--d_cell',
                           type=str, default='gru', choices=['gru'],
                           help='Decoder rnn cell type')
    model_arg.add_argument('--d_n_layers',
                           type=int, default=3,
                           help='Decoder number of layers')
    model_arg.add_argument('--d_dropout',
                           type=float, default=0,
                           help='Decoder layers dropout')
    model_arg.add_argument('--d_z',
                           type=int, default=128,
                           help='Latent vector dimensionality')
    model_arg.add_argument('--d_d_h',
                           type=int, default=512,
                           help='Decoder hidden dimensionality')
    model_arg.add_argument('--freeze_embeddings',
                           default=False, action='store_true',
                           help='If to freeze embeddings while training')

    # Train
    train_arg = parser.add_argument_group('Train')
    train_arg.add_argument('--n_batch',
                           type=int, default=512,
                           help='Batch size')
    train_arg.add_argument('--clip_grad',
                           type=int, default=50,
                           help='Clip gradients to this value')
    train_arg.add_argument('--kl_start',
                           type=int, default=0,
                           help='Epoch to start change kl weight from')
    train_arg.add_argument('--kl_w_start',
                           type=float, default=0,
                           help='Initial kl weight value')
    train_arg.add_argument('--kl_w_end',
                           type=float, default=0.05,
                           help='Maximum kl weight value')
    train_arg.add_argument('--lr_start',
                           type=float, default=3 * 1e-4,
                           help='Initial lr value')
    train_arg.add_argument('--lr_n_period',
                           type=int, default=10,
                           help='Epochs before first restart in SGDR')
    train_arg.add_argument('--lr_n_restarts',
                           type=int, default=10,
                           help='Number of restarts in SGDR')
    train_arg.add_argument('--lr_n_mult',
                           type=int, default=1,
                           help='Mult coefficient after restart in SGDR')
    train_arg.add_argument('--lr_end',
                           type=float, default=3 * 1e-4,
                           help='Maximum lr weight value')
    train_arg.add_argument('--n_last',
                           type=int, default=1000,
                           help='Number of iters to smooth loss calc')
    train_arg.add_argument('--n_workers',
                           type=int, default=1,
                           help='Number of workers for DataLoaders')
    
    return parser


def main(config):
    set_seed(config.seed)
    device = torch.device(config.device)

    if config.config_save is not None:
        torch.save(config, config.config_save)

    # For CUDNN to work properly
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device.index or 0)
    
    aff_data = pd.read_csv(pjoin('/mol_data', 'DeepAffinity', 'merged_data.tsv'), delimiter='\t')
    print(aff_data)
    embed_data = pd.read_csv(pjoin('/mol_data', 'embeds.csv'), delimiter='\t')
    train_data = (
        embed_data.values[10000:],
        aff_data['Sequence'].values[10000:],
        aff_data[['pEC50_[M]', 'pIC50_[M]', 'pKd_[M]', 'pKi_[M]']].values[10000:]
        )
    val_data = (
        embed_data.values[:10000],
        aff_data['Sequence'].values[:10000],
        aff_data[['pEC50_[M]', 'pIC50_[M]', 'pKd_[M]', 'pKi_[M]']].values[:10000],
        )
    trainer = BindingTrainer(config)

    vocab = trainer.get_vocabulary(train_data[1])

    if config.vocab_save is not None:
        torch.save(vocab, config.vocab_save)

    model = Binding(vocab, config).to(device)
    trainer.fit(model, train_data, val_data)

    model = model.to('cpu')
    torch.save(model.state_dict(), config.model_save)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    config.seed = 1

    config.model_save = get_model_path(config, 'binding')
    config.config_save = get_config_path(config, 'binding')
    config.vocab_save = get_vocab_path(config, 'binding')
    config.log_file = None
    main(config)
