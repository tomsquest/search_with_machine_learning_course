#!/usr/bin/env bash

set -euo pipefail
#set -x

MODEL_NAME="$1"
SOURCE_FILE="$2"
SHUFFLED_FILE="$1.shuf"

readonly COLOR_BLUE=$(tput bold)$(tput setaf 4)
readonly COLOR_RESET=$(tput sgr0)

function log {
    local MSG
    MSG="$(date +%T) $1"
    echo "$COLOR_BLUE$MSG$COLOR_RESET"
}

function createTrainAndTestFiles {
    log "Shuffling $SOURCE_FILE..."
    shuf --output "$SHUFFLED_FILE" "$SOURCE_FILE"
    log "Shuffled $SHUFFLED_FILE"

    local SPLIT="50000"
    log "Creating train/test files (with $SPLIT lines)"
    TRAIN_FILE="$SOURCE_FILE.train"
    TEST_FILE="$SOURCE_FILE.test"
    [ -f "$TRAIN_FILE" ] && rm "$TRAIN_FILE"
    [ -f "$TEST_FILE" ] && rm "$TEST_FILE"
    head -$SPLIT "$SHUFFLED_FILE" >"$TRAIN_FILE"
    tail -$SPLIT "$SHUFFLED_FILE" >"$TEST_FILE"
    log "- train: $TRAIN_FILE"
    log "- test:  $TEST_FILE"
}

function trainAndTest {
    # Hyperparameters
    local epochs=(25)
    local learningRates=(0.1)
    local wordNgrams=(2)
    # Evaluate these precisions
    local precisions=(1 3 5)

    for epoch in "${epochs[@]}"; do
        for lr in "${learningRates[@]}"; do
            for ngram in "${wordNgrams[@]}"; do
                # Train
                log "## Training epoch:$epoch lr:$lr wordNgrams:$ngram"
                fasttext supervised -input "$TRAIN_FILE" -output "$MODEL_NAME" -epoch "$epoch" -lr "$lr" -wordNgrams "$ngram"

                # Test
                for precision in "${precisions[@]}"; do
                    log "## Testing precision: $precision"
                    fasttext test "$MODEL_NAME.bin" "$TEST_FILE" "$precision"
                done
            done
        done
    done
}

function main {
    log "Starting"
    createTrainAndTestFiles
    trainAndTest
    log "Done"
}

main
