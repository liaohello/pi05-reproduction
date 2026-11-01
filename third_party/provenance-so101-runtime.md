# SO101 model runtime

Training/serving uses the pinned `openpi-server` image: `sha256:405af52051682d5e13169d1de1523fd0adc116eceae4fc314cadfacd45b1ad52`. Installed OpenPI source: 55 files, aggregate SHA-256 `d8ee60bcf03530a14986374a5a33e00915929368cfe5f9fa613ebbd8ee02c186`. A host `openpi-main` copy differed: hash `20a1b543a434fc2c4b3496253a4865754d4cb96a0c00323fbadaea027ff2a658`, 497,543 vs 497,539 bytes; it was not used.

Checkpoint `pi05_libero`: 16 files, 12,439,085,481 bytes, aggregate `7c81971fafcdbc677b0e8fd25b3bffd3d624abe4635323784f1918abdcef6f15`; parameter tree: 51 tensors, SHA-256 `bd475f88ac7e9e9134e1283ca1eff5001159701556578b48e1c68b5f97c33aa3`. PaliGemma tokenizer: 4,264,023 bytes, SHA-256 `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`.

Checkpoint/tokenizer are not redistributed. Tokenizer is copied to writable `OPENPI_DATA_HOME` for offline runs; image does not bundle it.
