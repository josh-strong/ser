from ser.params import Params, save_params
import tempfile
import shutil
import json
from pathlib import Path
from ser.constants import TEST_DIR

test_params_input = Params("test", 5, 4000, 0.01, "abcdefgh")
tmp_dirpath = tempfile.mkdtemp()
save_params(TEST_DIR / tmp_dirpath, test_params_input)


def test_save_params():
    test_params_input = Params("test", 5, 4000, 0.01, "abcdefgh")
    tmp_dirpath = tempfile.mkdtemp()
    save_params(tmp_dirpath, test_params_input)

    test_params_output = {
        "name": "test",
        "epochs": 1,
        "batch_size": 4000,
        "learning_rate": 0.01,
        "commit": "28029962b990777e52155d858213cc4dcfc38932",
    }

    with open(tmp_dirpath + "output.json") as fp:
        json.dump(test_params_output, fp)

    a = json.load(tmp_dirpath + "/params.json")
    b = json.load(tmp_dirpath + "/output.json")

    assert sorted(a.items()) == sorted(b.items())

    shutil.rmtree(tmp_dirpath)
