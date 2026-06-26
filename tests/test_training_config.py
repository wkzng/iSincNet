from training.trainer import TrainConfig


def test_train_config_spectrogram_scale_default_is_string():
    config = TrainConfig()

    assert config.spectrogram_scale == "mel"
    assert isinstance(config.spectrogram_scale, str)
