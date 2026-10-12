import fasttext

model = fasttext.load_model("lid.176.bin")

with open("Tatoeba.en-fr.fr", "r", encoding="utf-8") as f_fr, open("Tatoeba.en-fr.en", "r", encoding="utf-8") as f_en:
    fr_lines = f_fr.readlines()
    en_lines = f_en.readlines()

assert len(fr_lines) == len(en_lines), "Files must have the same number of lines."

filtered_fr = []
filtered_en = []

for fr, en in zip(fr_lines, en_lines):
    pred_fr = model.predict(fr.strip())[0][0]
    pred_en = model.predict(en.strip())[0][0]

    if pred_fr == '__label__fr' and pred_en == '__label__en':
        filtered_fr.append(fr)
        filtered_en.append(en)

with open("filtered_french.txt", "w", encoding="utf-8") as f_fr_out, open("filtered_english.txt", "w", encoding="utf-8") as f_en_out:
    f_fr_out.writelines(filtered_fr)
    f_en_out.writelines(filtered_en)

print(f"Filtered: kept {len(filtered_fr)} out of {len(fr_lines)} sentence pairs.")

