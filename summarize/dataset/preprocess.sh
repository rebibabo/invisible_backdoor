# !/usr/bin/env bash
# download dataset and preprocess
language=python
directory_path=${language}/final
if [ ! -d "$directory_path" ]; then
    wget https://zenodo.org/record/7857872/files/${language}.zip
    unzip "${language}".zip
    rm "${language}".zip
    rm "${language}"_licenses.pkl
    rm "${language}"_dedupe_definitions_v2.pkl
fi
python preprocess.py --lang ${language}
