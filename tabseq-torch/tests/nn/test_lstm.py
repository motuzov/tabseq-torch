import pytest
import torch
from tabseq_torch.nn import TabLSTM, PoolingType, Attention


@pytest.mark.parametrize("pooling_type", [PoolingType.AVG, PoolingType.LAST])
@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("attention", [False, True])
def test_TabLSTM(pooling_type, bidirectional, attention):
    input_size = 128
    h_size = 64
    num_classes = 2
    seq_len = 50
    batch_size = 30

    inputs = torch.ones(batch_size, seq_len, input_size)

    model = TabLSTM(
        input_size=input_size,
        h_size=h_size,
        num_classes=num_classes,
        pooling_type=pooling_type,
        attention=attention,
        bidirectional=bidirectional,
        num_layers=5,
    )

    outputs = model(inputs)
    assert outputs.shape == (batch_size, num_classes)


def test_att():
    batch = 3
    i_size = 10
    sl = 5
    h_size = 4
    in_ = torch.rand(batch, sl, i_size)
    num_layers = 2
    lstm = torch.nn.LSTM(
        input_size=i_size, hidden_size=h_size, num_layers=num_layers, batch_first=True
    )
    ho_t, _ = lstm(in_)
    att = Attention(h_size=h_size, bidirectional=False)
    actx = att(ho_t)
    assert actx.shape == torch.Size([3, 4])


def test_att_bedir():
    batch = 3
    i_size = 10
    sl = 5
    h_size = 4
    in_ = torch.rand(batch, sl, i_size)
    num_layers = 2
    bidirectional = True
    lstm_bd = torch.nn.LSTM(
        input_size=i_size,
        hidden_size=h_size,
        num_layers=num_layers,
        batch_first=True,
        bidirectional=bidirectional,
    )
    ho_t, _ = lstm_bd(in_)
    att = Attention(h_size=h_size, bidirectional=bidirectional)
    actx = att(ho_t)
    assert att.out_featurs == h_size * 2
    assert actx.shape == torch.Size([3, 8])
