from ser.transforms import flip
import torch

test_array_input = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
test_array_output = torch.tensor([[9, 8, 7], [6, 5, 4], [3, 2, 1]])


def test_flip():
    assert flip()(test_array_input).all() == test_array_output.all()
