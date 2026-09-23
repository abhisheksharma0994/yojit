#!/usr/bin/env bash
# Fetches the TLA+ tools jar used by formal/tla/*.tla.
#
# The jar is a 4.5 MB binary, so it is not committed (see .gitignore). Pinned to
# a specific GitHub release asset and verified against a known SHA-256: TLC's
# verdicts are only attributable to a toolchain if the bytes are known, and the
# asset for a release tag has been re-uploaded before now.
#
# The asset is `tla2tools.jar` on the v1.8.0 release; TLC self-reports its own
# version string, which is date-stamped rather than "1.8.0".
set -euo pipefail

RELEASE="v1.8.0"
URL="https://github.com/tlaplus/tlaplus/releases/download/${RELEASE}/tla2tools.jar"
EXPECTED_SHA256="9732eea90bdc7432e618184e4bee78700460e83e988238a80151dfd6507cfa0c"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
jar="${here}/tla2tools.jar"

# macOS ships shasum, most Linux images ship sha256sum; support both.
if command -v sha256sum >/dev/null 2>&1; then
  verify() { sha256sum -c - >/dev/null 2>&1; }
else
  verify() { shasum -a 256 -c - >/dev/null 2>&1; }
fi

if [ -f "$jar" ] && echo "${EXPECTED_SHA256}  ${jar}" | verify; then
  echo "tla2tools.jar already present and verified (TLA+ ${RELEASE})."
  exit 0
fi

echo "Downloading TLA+ tools from ${RELEASE}..."
curl -fsSL -o "${jar}.tmp" "$URL"
if ! echo "${EXPECTED_SHA256}  ${jar}.tmp" | verify; then
  rm -f "${jar}.tmp"
  echo "Checksum mismatch: the release asset is not the one this repo verified." >&2
  echo "If TLA+ re-uploaded the asset, re-verify with a known-good jar and update" >&2
  echo "EXPECTED_SHA256 here and the toolchain note in formal/README.md." >&2
  exit 1
fi
mv "${jar}.tmp" "$jar"
echo "Verified: ${jar}"
