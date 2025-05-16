import torch
import torch.nn as nn
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer


class GlobalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(GlobalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        repeated_hidden = hidden.unsqueeze(1).repeat(1, max_len, 1)
        # print(f"repeated_hidden shape: {repeated_hidden.shape}, encoder_outputs shape: {encoder_outputs.shape}")
        energy = torch.tanh(self.attn(torch.cat((repeated_hidden, encoder_outputs), dim=2)))
        # print(f"energy shape: {energy.shape}")
        attention_scores = self.v(energy).squeeze(2)
        attention_weights = nn.functional.softmax(attention_scores, dim=1)
        context_vector = (encoder_outputs * attention_weights.unsqueeze(2)).sum(dim=1)
        return context_vector


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(CrossAttention, self).__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.query_linear = nn.Linear(query_dim, query_dim)
        self.key_linear = nn.Linear(key_dim, query_dim)
        self.value_linear = nn.Linear(value_dim, query_dim)

    def forward(self, query, key, value):
        query_emb = self.query_linear(query)
        key_emb = self.key_linear(key)
        attention_weights = torch.bmm(query_emb, key_emb.transpose(1, 2))
        attention_weights = torch.softmax(attention_weights, dim=-1)

        value_emb = self.value_linear(value)
        attended_values = torch.bmm(attention_weights, value_emb)

        return attended_values

class TransformerBiLSTMGATTCrossAttModel(nn.Module):
    def __init__(self, batch_size, input_dim, output_size, num_layers,
                 num_heads, hidden_dim, hidden_layer_sizes, attention_dim, seq, seq_out, dropout):
        super(TransformerBiLSTMGATTCrossAttModel, self).__init__()
       
        self.batch_size = batch_size
        self.seq_out = seq_out
        self.output_size = output_size


        self.unsampling = nn.Conv1d(input_dim, hidden_layer_sizes[0], 1)

        self.hidden_dim = hidden_dim
        # Time Transformer layers
        self.timetransformer = TransformerEncoder(
            TransformerEncoderLayer(hidden_layer_sizes[0], num_heads, hidden_dim, dropout=dropout,
                                    batch_first=True),
            num_layers
        )


        self.hidden_layer_sizes = hidden_layer_sizes  
        self.num_layers = len(hidden_layer_sizes)
        self.bilstm_layers = nn.ModuleList() 

        self.bilstm_layers.append(
            nn.LSTM(input_dim, hidden_layer_sizes[0], batch_first=True, bidirectional=True))  # 7→32

        for i in range(1, self.num_layers):
            self.bilstm_layers.append(
                nn.LSTM(hidden_layer_sizes[i - 1] * 2, hidden_layer_sizes[i], batch_first=True,
                        bidirectional=True))  # 64→64


        self.globalAttention = GlobalAttention(attention_dim * 2)  # 双向LSTM 维度 *2  128

        self.cross_attention = CrossAttention(hidden_layer_sizes[0], hidden_layer_sizes[-1] * 2,
                                              hidden_layer_sizes[-1] * 2)


        self.conv_feature = nn.Conv1d(hidden_layer_sizes[0], output_size, kernel_size=3, padding=1, stride=1)
        self.conv_seq = nn.Conv1d(seq, seq_out, kernel_size=3, padding=1, stride=1)


        self.fc = nn.Linear(hidden_layer_sizes[0], output_size)

    def forward(self, input_seq):

        # print(input_seq.shape)
        unsampling = self.unsampling(input_seq.permute(0, 2, 1))
        transformer_output = self.timetransformer(unsampling.permute(0, 2, 1))


        bilstm_out = input_seq
        for bilstm in self.bilstm_layers:
            bilstm_out, (hidden, cell) = bilstm(bilstm_out)


        # print(f"hidden shape: {hidden.shape}, bilstm_out shape: {bilstm_out.shape}")


        hidden_concat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        gatt_features = self.globalAttention(hidden_concat, bilstm_out)
        gatt_features = gatt_features.reshape(input_seq.size(0), -1,
                                              self.hidden_layer_sizes[-1] * 2)

        query = transformer_output
        key = gatt_features
        value = gatt_features
        cross_attention_features = self.cross_attention(query, key, value)


        # print(f'cross_attention_features: {cross_attention_features.shape}')
        cross_attention_features = cross_attention_features.permute(0, 2, 1)  # (batch, feature, seq_len)
        # print(f'cross_attention_features: {cross_attention_features.shape}')
        adjusted_features = self.conv_feature(cross_attention_features)  # (batch, feature, seq_out)
        # print(f'adjusted_features: {adjusted_features.shape}')
        adjusted_seq_len = adjusted_features.permute(0, 2, 1)  # (batch, seq_out, feature)
        predict = self.conv_seq(adjusted_seq_len)
        # print(f'predict: {predict.shape}')

        # predict = self.fc(adjusted_features)  # 输出维度为[batch_size, seq_out, output_size]
        # print(predict.shape)
        return predict
