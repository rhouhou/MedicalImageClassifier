from src.data import build_transforms

def test_build_transforms():
    t_train = build_transforms(224, augment=True)
    t_eval = build_transforms(224, augment=False)
    assert t_train is not None and t_eval is not None