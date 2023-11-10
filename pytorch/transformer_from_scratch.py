import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = self.embed_size // self.num_heads

        assert (self.head_dim * self.num_heads ==
                self.embed_size), "Embed size needs to be divisible by num_heads."

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(
            self.num_heads * self.head_dim, self.embed_size)

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split embedding into self.num_heads pieces
        # Shape: [N, value_len, num_heads, head_dim]
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        # Shape: [N, key_len, num_heads, head_dim]
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        # Shape: [N, query_len, num_heads, head_dim]
        queries = queries.reshape(N, query_len, self.num_heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Shape: [N, heads, query_len, key_len]
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Shape: [N, heads, query_len, key_len]
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        # Shape: [N, query_len, num_heads, head_dim]
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_dim)

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out


class Encoder(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 embed_size,
                 num_layers,
                 num_heads,
                 device,
                 forward_expansion,
                 dropout,
                 max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, num_heads,
                                 dropout, forward_expansion)
                for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(
            N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(
            x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, num_heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, num_heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, source_mask, target_mask):
        # value and key come from encoder output.
        attention = self.attention(x, x, x, target_mask)
        query = self.dropout(self.norm(attention + x))

        out = self.transformer_block(value, key, query, source_mask)
        return out


class Decoder(nn.Module):
    def __init__(self,
                 target_vocab_size,
                 embed_size,
                 num_layers,
                 num_heads,
                 forward_expansion,
                 dropout,
                 device,
                 max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, num_heads,
                             forward_expansion, dropout, device)
                for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, source_mask, target_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(
            N, seq_length).to(self.device)

        x = self.dropout(self.word_embedding(
            x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(x, encoder_output, encoder_output,
                        source_mask, target_mask)

        out = torch.softmax(self.fc_out(out), dim=2)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        source_vocab_size,
        target_vocab_size,
        source_pad_index,
        target_pad_index,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        num_heads=8,
        dropout=0.0,
        device="cuda",
        max_length=100
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(source_vocab_size, embed_size, num_layers,
                               num_heads, device, forward_expansion, dropout, max_length)
        self.decoder = Decoder(target_vocab_size, embed_size, num_layers,
                               num_heads, forward_expansion, dropout, device, max_length)

        self.source_pad_index = source_pad_index
        self.target_pad_index = target_pad_index
        self.device = device

    def make_source_mask(self, source):
        # Shape: [N, 1, 1, source_len]
        source_mask = (source != self.source_pad_index).unsqueeze(
            1).unsqueeze(2)
        return source_mask.to(self.device)

    def make_target_mask(self, target):
        N, target_len = target.shape
        target_mask = torch.tril(torch.ones((target_len, target_len))).expand(
            N, 1, target_len, target_len)
        return target_mask.to(self.device)

    def forward(self, source, target):
        source_mask = self.make_source_mask(source)
        target_mask = self.make_target_mask(target)
        encoder_source = self.encoder(source, source_mask)
        out = self.decoder(target, encoder_source, source_mask, target_mask)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}...")

    x = torch.tensor(
        [[1, 5, 6, 4, 3, 9, 5, 2, 0],
         [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    target = torch.tensor(
        [[1, 7, 4, 3, 5, 9, 2, 0],
         [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    source_pad_index = 0
    target_pad_index = 0
    source_vocab_size = 10
    target_vocab_size = 10
    model = Transformer(source_vocab_size, target_vocab_size,
                        source_pad_index, target_pad_index, device=device).to(device)
    out = model(x, target[:, :-1])
    print(out.shape)
    print(out)
