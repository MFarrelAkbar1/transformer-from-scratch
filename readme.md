# Implementasi Transformer dari Nol

**Muhammad Farrel Akbar (22/492806/TK/53947)**

Implementasi arsitektur GPT-style Transformer menggunakan NumPy tanpa framework deep learning.

## Deskripsi

Proyek ini mengimplementasikan komponen inti Transformer untuk autoregressive language modeling, termasuk:

- Token embedding dengan scaling
- Sinusoidal positional encoding
- Scaled dot-product attention
- Multi-head attention (8 heads)
- Feed-forward network dengan aktivasi GELU
- Layer normalization (pre-norm architecture)
- Residual connections
- Causal masking untuk autoregressive generation

## Dependensi

```bash
numpy>=1.20.0
matplotlib>=3.3.0
```

## Instalasi

```bash
pip install numpy matplotlib
```

## Penggunaan

Jalankan notebook Jupyter:

```bash
jupyter notebook gpt-example.ipynb
```

Atau konversi ke script Python:

```bash
jupyter nbconvert --to script gpt-example.ipynb
python gpt-example.py
```

## Struktur Kode

- **Cell 1**: Token Embedding
- **Cell 2**: Positional Encoding
- **Cell 3**: Scaled Dot-Product Attention
- **Cell 4**: Multi-Head Attention
- **Cell 5**: Feed-Forward Network
- **Cell 6**: Layer Normalization dan Residual Connection
- **Cell 7-8**: Causal Masking
- **Cell 9**: Output Layer
- **Cell 10-11**: Model Lengkap dan Testing
- **Cell 12-14**: Fitur Bonus (Weight Tying, Visualisasi, Perbandingan PE)

## Konfigurasi Model

```python
vocab_size = 1000
d_model = 256
num_heads = 8
d_ff = 1024
num_layers = 3
max_seq_len = 512
```

## Testing

Program mencakup pengujian untuk:
- Validasi dimensi tensor
- Verifikasi properti softmax
- Validasi causal masking
- Uji integrasi model lengkap