[Controlled Generation of Synthetic Speaker Vectors for Voice Anonymization]

Source code for the paper.

# Abstract

Voice anonymization benefits from transparent control over the pseudo-speaker's voice. Anonymization approaches based on resynthesis substitute the original speaker vector with a different, often artificial, vector, but this does not allow for controlling which original speaker attributes are preserved. To address this, we extend one existing WGAN-based artificial vector generation method with attribute-label conditioning. We also introduce diffusion models as a robust alternative for generating speaker vectors, utilizing classifier guidance for label-informed synthesis. Evaluated using the Voice Privacy Challenge 2024 suite, both approaches demonstrate strong attribute control while maintaining competitive privacy and utility. Furthermore, we propose measures for diversity, originality, and naturalness of synthesized embeddings. Combined with TTS performance on challenging speech, these measures enable the selection of optimal generator configurations for the anonymization pipeline.


This work is incremental to and builds upon [DigitalPhonetics/speaker-anonymization](https://github.com/DigitalPhonetics/speaker-anonymization).

# Status

🚧 This repository is under active construction. Results, usage instructions, and pretrained models will be added soon.