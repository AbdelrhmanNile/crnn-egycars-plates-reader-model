ar_to_en = {"١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9", "٠": "0",
            "أ": "A", "ب": "B", "ج": "G", "ح": "H", "د": "D", "ر": "R", "س": "C", "ص": "S", "ط": "T", "ع": "E", "ف": "F", "ق": "Q", "ك": "K", "ل": "L", "م": "M", "ن": "N", "ه": "O", "و": "W", "ى": "Y"}

CHAR_VECTOR = "ABCDEFGHKLMNOQRSTWY0123456789X"
letters = [letter for letter in CHAR_VECTOR]

num_classes = len(letters) + 1

img_w, img_h = 128, 64

# Network parameters
batch_size = 8
val_batch_size = 1

downsample_factor = 4
max_text_len = 7
