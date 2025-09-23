#!/usr/bin/env bash
set -euo pipefail

SR=16000           # sample rate
CH=1               # channels (mono)
ACODEC="pcm_s16le" # 16-bit PCM
OUTDIR="."
OVERWRITE="-n" # don't overwrite by default
RECURSIVE=0

usage() {
    cat <<'EOF'
Usage:
  convert_audio.sh [options] <file|dir> [more files...]

Options:
  -r <rate>     Sample rate (default: 16000)
  -c <ch>       Channels (default: 1)
  -d <codec>    Audio codec (default: pcm_s16le; e.g., pcm_f32le)
  -o <outdir>   Output directory (default: .)
  -f            Force overwrite existing files
  -R            If an argument is a directory, convert files in it recursively
  -h            Help

Examples:
  convert_audio.sh input.mp3
  convert_audio.sh -o out -r 16000 -c 1 song.flac podcast.m4a
  convert_audio.sh -R -o out ./my_audio_folder
EOF
}

detect_pm() {
    case "$(uname -s)" in
    Darwin) command -v brew >/dev/null 2>&1 && {
        echo "brew"
        return
    } ;;
    Linux)
        for pm in apt dnf yum pacman zypper apk; do
            command -v "$pm" >/dev/null 2>&1 && {
                echo "$pm"
                return
            }
        done
        ;;
    MINGW* | MSYS* | CYGWIN* | Windows_NT)
        command -v winget >/dev/null 2>&1 && {
            echo "winget"
            return
        }
        command -v choco >/dev/null 2>&1 && {
            echo "choco"
            return
        }
        ;;
    esac
    echo ""
}

ensure_ffmpeg() {
    if command -v ffmpeg >/dev/null 2>&1; then return; fi
    echo "ffmpeg not found."
    PM=$(detect_pm)
    if [[ -n "$PM" ]]; then
        read -r -p "Install ffmpeg via $PM? (Y/n) " yn
        yn=${yn:-Y}
        if [[ "$yn" =~ ^[Yy]$ ]]; then
            case "$PM" in
            brew) brew install ffmpeg ;;
            apt) sudo apt update && sudo apt install -y ffmpeg ;;
            dnf) sudo dnf install -y ffmpeg ;;
            yum) sudo yum install -y ffmpeg ;;
            pacman) sudo pacman -Sy --noconfirm ffmpeg ;;
            zypper) sudo zypper install -y ffmpeg ;;
            apk) sudo apk add --no-cache ffmpeg ;;
            winget) winget install --id=Gyan.FFmpeg -e ;;
            choco) choco install ffmpeg -y ;;
            *)
                echo "Unsupported package manager: $PM"
                exit 1
                ;;
            esac
        else
            echo "Please install ffmpeg and re-run."
            exit 1
        fi
    else
        echo "No package manager detected. Please install ffmpeg and re-run."
        exit 1
    fi
}

convert_one() {
    local in="$1"
    local base name ext out
    name="$(basename -- "$in")"
    ext="${name##*.}"
    base="${name%.*}"
    mkdir -p "$OUTDIR"
    out="$OUTDIR/$base.wav"

    # Choose overwrite behavior
    local owflag="$OVERWRITE" # -n (no overwrite) or -y (force)

    echo "Converting: $in -> $out"
    ffmpeg -nostdin -hide_banner -loglevel error \
        -i "$in" \
        -ac "$CH" -ar "$SR" -c:a "$ACODEC" \
        $owflag "$out"
}

# Collect audio-like files in a directory (by extension)
gather_from_dir() {
    local dir="$1"
    if [[ "$RECURSIVE" -eq 1 ]]; then
        find "$dir" -type f \( \
            -iname '*.mp3' -o -iname '*.m4a' -o -iname '*.aac' -o -iname '*.ogg' -o -iname '*.oga' -o \
            -iname '*.opus' -o -iname '*.flac' -o -iname '*.wav' -o -iname '*.aiff' -o -iname '*.aif' -o \
            -iname '*.wma' -o -iname '*.amr' -o -iname '*.webm' -o -iname '*.mka' -o -iname '*.3gp' -o \
            -iname '*.mp4' -o -iname '*.mov' \
            \)
    else
        find "$dir" -maxdepth 1 -type f \( \
            -iname '*.mp3' -o -iname '*.m4a' -o -iname '*.aac' -o -iname '*.ogg' -o -iname '*.oga' -o \
            -iname '*.opus' -o -iname '*.flac' -o -iname '*.wav' -o -iname '*.aiff' -o -iname '*.aif' -o \
            -iname '*.wma' -o -iname '*.amr' -o -iname '*.webm' -o -iname '*.mka' -o -iname '*.3gp' -o \
            -iname '*.mp4' -o -iname '*.mov' \
            \)
    fi
}

# Parse flags
while getopts ":r:c:d:o:fRh" opt; do
    case $opt in
    r) SR="$OPTARG" ;;
    c) CH="$OPTARG" ;;
    d) ACODEC="$OPTARG" ;; # e.g., pcm_f32le
    o) OUTDIR="$OPTARG" ;;
    f) OVERWRITE="-y" ;;
    R) RECURSIVE=1 ;;
    h)
        usage
        exit 0
        ;;
    \?)
        echo "Unknown option: -$OPTARG"
        usage
        exit 1
        ;;
    :)
        echo "Option -$OPTARG requires an argument."
        usage
        exit 1
        ;;
    esac
done
shift $((OPTIND - 1))

# Require at least one path
if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

ensure_ffmpeg

# Process args (files or dirs)
for arg in "$@"; do
    if [[ -d "$arg" ]]; then
        mapfile -t files < <(gather_from_dir "$arg")
        if [[ ${#files[@]} -eq 0 ]]; then
            echo "No audio files found in: $arg"
            continue
        fi
        for f in "${files[@]}"; do convert_one "$f"; done
    elif [[ -f "$arg" ]]; then
        convert_one "$arg"
    else
        echo "Skip: $arg (not found)"
    fi
done

echo "Done."
