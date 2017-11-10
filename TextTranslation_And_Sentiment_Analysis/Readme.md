# Description
This project is to convert a provided language to english language and run the vader sentiment analysis on it.

## Installation
Use `pip`:

`pip install yandex.translate`

`pip install ntlk`

## Usage

import translate_py
#the following code with import
arr=translate_py.translate_language(chineese_text,'zh','en')
SentimentAnalyzer('Line',arr)
