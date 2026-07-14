# Low-disk update behavior

Metaflora Incubus downloads two signed release artifacts: a small platform runtime archive and a direct GGUF file. Both URLs, immutable revisions, byte sizes, and SHA-256 digests are pinned by the signed release manifest. The installer does not build or convert model weights on the user's computer.

A fresh installation is limited to a 5 GiB peak-disk budget, including a 256 MiB safety reserve. The GGUF is downloaded once, directly into a staging directory on the final filesystem. It is not wrapped in a second model archive and is not copied again during activation.

## Safe update tradeoff

`incubusctl update` never deletes the active GGUF to make room. It keeps the complete old installation until the replacement has been downloaded, validated, activated, and passed its health check. A failed update can therefore restore the working release.

This safe update temporarily needs room for both the installed GGUF and the new split release. When that space is unavailable, the command stops before downloading or changing anything and instructs the user to run `incubusctl uninstall`, then install the new release. That explicit path sacrifices automatic rollback, but it never silently destroys the working model in the middle of an update.

## Signed manifest schema

Release manifests use schema version 2. Each artifact records an immutable 40-character Hugging Face commit, the exact URL containing that commit, its compressed or direct byte size, SHA-256 digest, and Ed25519 signature.

The manifest contains exactly one `model` artifact with `format: "gguf"`. It also contains one `runtime` artifact per supported platform with `format: "tar.gz"` and `unpacked_size_bytes`. The installer rejects a release when:

- either half of the split release is missing;
- a revision is mutable or does not match its URL;
- a digest, signature, or byte size differs;
- runtime extraction exceeds the signed unpacked-size limit;
- model size, runtime download, runtime extraction, and the 256 MiB reserve exceed 5 GiB in total;
- the host does not actually have the calculated peak space available.
