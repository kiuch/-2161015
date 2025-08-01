import MeCab

mecab = MeCab.Tagger()
result = mecab.parse("これはMeCabを動かすサンプルコードです。")

print(result)