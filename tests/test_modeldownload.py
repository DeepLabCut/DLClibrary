import pytest

def test_catdownload():
    # TODO: just download the lightweight stuff..
    import dlclibrary, os

    os.mkdir('cat')
    dlclibrary.download_hugginface_model('full_cat','cat')


    assert os.path.exists("cat/pose_cfg.yaml")
    assert os.path.exists("cat/snapshot-75000.meta")

    import shutil
    shutil.rmtree("cat")
