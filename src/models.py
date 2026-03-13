"""Backward compatibility shim — imports from neuromorph.models."""
from neuromorph.models.encoder import Encoder
from neuromorph.models.decoder import Decoder
from neuromorph.models.autoencoder import Autoencoder
from neuromorph.models.utils import freeze_except_context, unfreeze_all

__all__ = ['Encoder', 'Decoder', 'Autoencoder', 'freeze_except_context', 'unfreeze_all']
