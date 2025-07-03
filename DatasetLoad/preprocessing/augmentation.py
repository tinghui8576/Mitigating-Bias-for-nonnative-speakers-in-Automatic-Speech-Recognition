from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# Audio augmentations:
AUGMENT = Compose([
    AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=0.2),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.2, leave_length_unchanged=False),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
    # SpecAugment(time_mask_param=30, freq_mask_param=15, p=0.3)
    ])

def augment_audio(batch, sample_rate: int):
    """
    Perform data augmentation for audio.
    """
    audio = batch["audio"]["array"]
    augmented_audio = AUGMENT(samples=audio, sample_rate=sample_rate)

    batch["audio"]["array"] = augmented_audio

    return batch