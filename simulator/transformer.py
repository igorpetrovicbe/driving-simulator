import torch
import torch.nn as nn


class MultiLayerTransformer(nn.Module):
    def __init__(self, d_model, vocab_size, ff_size, num_layers, num_heads, max_input_sequence_length,
                 max_output_sequence_length):
        super(MultiLayerTransformer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_input_sequence_length = max_input_sequence_length
        self.max_output_sequence_length = max_output_sequence_length

        self.embedding = nn.Embedding(vocab_size, d_model)

        # Define the Transformer layer with multiple layers
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ff_size,
                                                      batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)

        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ff_size,
                                                      batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=num_layers)

        # Define the output layer
        self.out_fc = nn.Linear(d_model, vocab_size - 1)

    def forward(self, input, context):
        # Add positional encodings to the input
        input_positional_encodings = self.create_positional_encoding(input.size(0), input.size(1), self.d_model)
        input = self.embedding(input)
        input = input + input_positional_encodings.to(input.device)

        # Add positional encodings to the context
        context_positional_encodings = self.create_positional_encoding(context.size(0), context.size(1), self.d_model)
        context = self.embedding(context)
        # Ensure both tensors are on the same device and have the same shape and dtype
        context_positional_encodings = context_positional_encodings.to(input.device)
        context = context + context_positional_encodings

        # Generate causal mask
        mask_context = nn.Transformer.generate_square_subsequent_mask(context.size(1)).to(context.device)

        # Forward pass through the Transformer layer with attention mask
        encoder_out = self.transformer_encoder(input)
        decoder_out = self.transformer_decoder(context, memory=encoder_out, tgt_mask=mask_context, tgt_is_causal=True)

        # Forward pass through the output layer
        output = self.out_fc(decoder_out)

        return output

    def encode(self, input):
        # Add positional encodings to the input
        input_positional_encodings = self.create_positional_encoding(input.size(0), input.size(1), self.d_model)
        input = self.embedding(input)
        input = input + input_positional_encodings.to(input.device)

        encoder_out = self.transformer_encoder(input)
        return encoder_out

    def decode(self, encoder_out, context):
        # Add positional encodings to the context
        context_positional_encodings = self.create_positional_encoding(context.size(0), context.size(1), self.d_model)
        context = self.embedding(context)
        # Ensure both tensors are on the same device and have the same shape and dtype
        context_positional_encodings = context_positional_encodings.to(encoder_out.device)
        context = context + context_positional_encodings

        # Generate causal mask
        mask_context = nn.Transformer.generate_square_subsequent_mask(context.size(1)).to(context.device)

        decoder_out = self.transformer_decoder(context, memory=encoder_out, tgt_mask=mask_context, tgt_is_causal=True)

        # Forward pass through the output layer
        output = self.out_fc(decoder_out)

        return output

    def create_positional_encoding(self, batch_size, max_len, d_model):
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pos_enc = torch.zeros((max_len, d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0).expand(batch_size, -1, -1)  # Add batch dimension
        return pos_enc
