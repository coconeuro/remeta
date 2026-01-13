from remeta_v1 import remeta

# x_stim, d_dec, c_conf, params, y_decval = remeta.load_dataset(
#     'noisy_readout', return_params=True, return_y_decval=True
# )
# remeta.plot_evidence_versus_confidence(x_stim, c_conf, y_decval, params, plot_bias_free=True)


x_stim, d_dec, c_conf, params, y_decval = remeta.load_dataset(
    'noisy_readout', return_params=True, return_y_decval=True
)
cfg = remeta.Configuration()
cfg.type2_noise_type = 'noisy_readout'
cfg.gridsearch = False
cfg.true_params = params
rem = remeta.ReMeta(cfg)

rem.fit(x_stim, d_dec, c_conf)