from .rnn import *

def SC1D_model(args) :
    if args.model_name == 'RNN' :
        from models.SC1D_models.rnn import RNN
        return RNN(input_dim=args.num_channels, hidden_dim=64, num_classes=args.num_classes, dropout_prob=0.5)
