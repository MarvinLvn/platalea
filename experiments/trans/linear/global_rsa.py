import platalea.analysis.phoneme as P

config = dict(directory = '../../../data/out/trans/',
              attention = 'linear',
              attention_hidden_size = None,
              hidden_size = None,
              epochs = 60,
              test_size = 1/2,
              #layers=['conv2', 'convout'] + [ 'transf{}'.format(i) for i in range(12) ]
              layers=['convout'] + [ 'transf{}'.format(i) for i in range(12) ],
              device = 'cuda:1'
              )

P.global_rsa(config)
P.global_rsa_plot()
