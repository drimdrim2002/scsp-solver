#!/usr/bin/env bash
# Download the benchmark instances into benchmarks/instances/ (not tracked by git).
#
#   scripts/download_instances.sh [scp] [spp] [miplib]      (default: all three)
#
# Sources
#   OR-Library (J.E. Beasley): http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/
#   MIPLIB:                    https://miplib.zib.de/WebData/instances/
# Files that exist as .gz on the server are fetched as .gz and kept compressed;
# the readers open .gz transparently.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/benchmarks/instances"
ORLIB="http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files"
MIPLIB="https://miplib.zib.de/WebData/instances"

fetch() {  # fetch <url> <dest-file>
  local url="$1" out="$2"
  if [[ -s "$out" ]]; then return 0; fi
  curl --fail --silent --show-error --location --retry 3 --max-time 600 -o "$out.part" "$url"
  mv "$out.part" "$out"
  echo "ok  $(basename "$out")"
}

scp_files() {
  local i s
  for i in 1 2 3 4 5 6 7 8 9 10; do echo "scp4$i.txt"; echo "scp5$i.txt"; done
  for i in 1 2 3 4 5; do echo "scp6$i.txt"; done
  for s in a b c d e; do for i in 1 2 3 4 5; do echo "scp$s$i.txt"; done; done
  for s in e f g h; do for i in 1 2 3 4 5; do echo "scpnr$s$i.gz"; done; done
  echo rail507.gz; echo rail516.gz; echo rail582.gz
}

spp_files() {
  local i n
  for i in $(seq 1 43); do
    n=$(printf "%02d" "$i")
    case "$n" in
      01|02|03|04|05|14|16|17) echo "sppnw$n.gz" ;;
      *) echo "sppnw$n.txt" ;;
    esac
  done
  for i in 1 2 3 4 5 6; do echo "sppaa0$i.txt"; done
  echo sppus01.gz; echo sppus02.txt; echo sppus03.gz; echo sppus04.gz
  echo sppkl01.txt; echo sppkl02.txt
}

sets=("$@")
if [[ ${#sets[@]} -eq 0 ]]; then sets=(scp spp miplib); fi

for s in "${sets[@]}"; do
  case "$s" in
    scp)
      mkdir -p "$DEST/orlib_scp"
      for f in $(scp_files); do fetch "$ORLIB/$f" "$DEST/orlib_scp/$f"; done ;;
    spp)
      mkdir -p "$DEST/orlib_spp"
      for f in $(spp_files); do fetch "$ORLIB/$f" "$DEST/orlib_spp/$f"; done ;;
    miplib)
      mkdir -p "$DEST/miplib"
      for f in air04 air05 nw04; do fetch "$MIPLIB/$f.mps.gz" "$DEST/miplib/$f.mps.gz"; done ;;
    *) echo "unknown set: $s" >&2; exit 2 ;;
  esac
done
echo "done: $(find "$DEST" -type f | wc -l | tr -d ' ') files, $(du -sh "$DEST" | cut -f1)"
