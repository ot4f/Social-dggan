from models import generator, discriminator
import sys
import config


# config
window_size = config.obs_seq_len
pred_len = config.pred_seq_len
in_features = config.in_features
out_features = config.out_features
lstm_features = config.lstm_features
disc_hidden = config.disc_hidden
out_size = config.output_size
max_node_num = config.max_node_num
disc_inpsize = config.disc_inpsize


# models
generators = {
    'cnn-gat': generator.CNN_GAT_Generator,
}

discriminators = {
    'lstm': discriminator.LSTM_Discriminator
}

# args
generator_base_args = {
    'window_size': window_size,
    'n_pred': pred_len,
    'in_features': in_features,
    'out_features': out_features,
    'out_size': out_size,
    'embedding_dim': 64,
    'n_stgcnn': config.n_stgcnn,
    'n_txpcnn': config.n_txpcnn,
    'node_num': max_node_num,
    'lstm_features': lstm_features
}

discriminator_base_args = {
    'input_size': pred_len * disc_inpsize,
    'hidden_size': disc_hidden
}

generator_args = {
    'cnn-gat': {**generator_base_args},
}

discriminator_args = {
    'lstm': {'input_size': disc_inpsize, 'embedding_size': 64, 'hidden_size': 64, 'mlp_size': 1024}
}


def get_model(model_name, model_config=None):
    """
    :param model_name
    :param args
    """
    if model_config is None:
        model_config = config.model_config
    print(model_config)
    genn = model_config['gen']
    discn = model_config['disc']
    if genn not in generators or discn not in discriminators:
        print('Model "%s" does not exist !!!' % model_name, genn, discn)
        sys.exit(1)
    else:
        generator_ = generators[genn](**generator_args[genn])
        discriminator_ = discriminators[discn](**discriminator_args[discn]) 
        
    return generator_, discriminator_