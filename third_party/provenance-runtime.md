# Image source and runtime

External images provide source; neither image contents nor private registry details are published.

| Image | Digest | Tree | Files | Bytes | Aggregate SHA-256 |
|---|---|---|---:|---:|---|
| openpi-server | `sha256:405af52051682d5e13169d1de1523fd0adc116eceae4fc314cadfacd45b1ad52` | OpenPI source tree | 140 | 1,606,857 | `e9a694731905fb9d22a56ca5267d71a8…` |
| openpi-client | `sha256:5dc5543f91a61a109181de1e1f1a50e771e40aaa1058207d1d5a3fab2bfb377a` | LIBERO source/assets tree | 1,116 | 443,644,276 | `e6d16c4a17b0a56708bb550273b321ef…` |

Aggregate is SHA-256 over sorted `relative_path\0size\0sha256` lines joined by newline. The client LIBERO commit is `8f1084e3132a39270c3a13ebe37270a43ece2a01`. OpenPI came from a release archive, so no verifiable git commit exists.

Acquisition and versions: [fingerprinting commands](provenance-runtime-commands.md).
