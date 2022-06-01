#!/usr/bin/env bash

DIR=data/molecules/
mkdir -p $DIR
cd $DIR


FILE=molecules.zip
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl "https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1" -o molecules.zip -J -L -k
	unzip molecules.zip -d ../
fi



