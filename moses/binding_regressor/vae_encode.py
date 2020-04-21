import argparse
import os
import sys

pjoin = os.path.join
up_dir = os.path.dirname
CURR_DIR = up_dir(os.path.realpath(__file__))
BASE_DIR = up_dir(up_dir(CURR_DIR))
sys.path.insert(1, BASE_DIR)
import torch
import torch.nn as nn
import rdkit
import pandas as pd
from tqdm.auto import tqdm
from moses.vae import VAE
from moses.models_storage import ModelsStorage
from moses.script_utils import add_sample_args, set_seed

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

MODELS = ModelsStorage()


class VAEEncode(VAE):
    def forward_encoder_no_noise(self, x):
        """Encoder step, emulating z ~ E(x) = q_E(z|x)

        :param x: list of tensors of longs, input sentence x
        :return: (n_batch, d_z) of floats, sample of latent vector z
        :return: float, kl term component of loss
        """
        with torch.no_grad():
            x = [self.x_emb(i_x) for i_x in x]
            x = nn.utils.rnn.pack_sequence(x)
            
            _, h = self.encoder_rnn(x, None)
            
            h = h[-(1 + int(self.encoder_rnn.bidirectional)):]
            h = torch.cat(h.split(1), dim=-1).squeeze(0)
            
            mu = self.q_mu(h)
            return mu


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1)
    parser.add_argument('--device', default=1)
    return parser


def main(config):
    set_seed(config.seed)
    device = torch.device(config.device)
    
    # For CUDNN to work properly:
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device.index or 0)
    
    # Hardcode
    model_config = torch.load(pjoin(BASE_DIR, 'checkpoints', 'vae', 'vae_1', 'vae_config.pt'))
    model_vocab = torch.load(pjoin(BASE_DIR, 'checkpoints', 'vae', 'vae_1', 'vae_vocab.pt'))
    model_state = torch.load(pjoin(BASE_DIR, 'checkpoints', 'vae', 'vae_1', 'vae_model.pt'))
    
    model = VAEEncode(model_vocab, model_config)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
    
    smile = pd.read_csv(pjoin('/mol_data', 'DeepAffinity', 'merged_data.tsv'), delimiter='\t')
    smile = smile['Canonical SMILE'].values
    embeds = []
    for smi in tqdm(smile, desc='Running VAE encoder'):
        smi = model.string2tensor(smi)
        embeds.append(model.forward_encoder_no_noise([smi]).cpu().numpy()[0])
    samples = pd.DataFrame(embeds)
    samples.to_csv(pjoin(BASE_DIR, 'checkpoints', 'vae', 'vae_1', 'embeds.csv'), index=False)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    main(config)
