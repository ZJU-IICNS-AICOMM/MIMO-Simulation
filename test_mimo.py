# -*- coding:utf-8 -*-
# @Time: 2023/6/1 15:49

import numpy as np
from mimo import MIMO_Channel
from qam_modulator import qam_mod, qam_mapper, draw_trx_constellation

# Settings
np.random.seed(seed=0)
Nt = 8  # number of transmitting antennas
K = 1  # number of users
Nr = 8  # number of receiving antennas
d = 2  # data streams, d <= min(Nt/K, Nr)
P = 1  # power constraint
M = 64  # modulation order
snr = 10  # signal-to-noise ratio
snr_range = np.arange(0, 50, 0.1)  # test SNR range

# M-QAM Modulation
tx_bits = np.random.randint(0, 2, 40000)
mapping_table, demapping_table = qam_mod(M)
tx_symbols = qam_mapper(tx_bits, mapping_table)
draw_trx_constellation(tx_symbols, tx=True, snr=snr, channel='mimo')
# MIMO Channel
mimo_channel = MIMO_Channel(Nr, Nt, d, K, P)
# rx_symbols = mimo_channel.circular_gaussian(tx_symbols, snr)
rx_symbols = mimo_channel.mmwave_MIMO(tx_symbols, snr)
draw_trx_constellation(rx_symbols, tx=False, snr=snr, channel='mimo')
print('Completed.')
