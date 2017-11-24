#!/usr/bin/env bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Temporarily setting the internal field seperator (IFS) to the newline character.
IFS=$'\n';

# Recursively loop through all WAV files in the specified directory
WAV_DIRECTORY=$1
for wav in $(find ${WAV_DIRECTORY} -name '*.wav'); do
	echo "Processing WAV file ${wav}"

	INPUT_WAV_FILE="${wav}"
	OUTPUT_OGG_FILE="${wav%.wav}.ogg"

	if [ -f $OUTPUT_OGG_FILE ]; then
		echo "Output OGG file ${OUTPUT_OGG_FILE} already found"
		echo "Skipping conversion for WAV file ${INPUT_WAV_FILE}"
		continue
	fi

	INPUT_WAV_DIR=$(dirname "${INPUT_WAV_FILE}")
	cd ${INPUT_WAV_DIR}

	# NOTE: metadata is removed
	ffmpeg -i ${INPUT_WAV_FILE} -map_metadata -1 -acodec libvorbis -ac 1 -ar 16000 -aq 5 ${OUTPUT_OGG_FILE}
	if ! [ -f $OUTPUT_OGG_FILE ]; then
		echo "Could not find output OGG file ${OUTPUT_OGG_FILE}. An error probably occured during conversion."
		exit 1
	fi
done

echo 'All done.'
