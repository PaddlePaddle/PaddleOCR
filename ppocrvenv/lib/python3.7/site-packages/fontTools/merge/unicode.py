# Copyright 2021 Behdad Esfahbod. All Rights Reserved.

def is_Default_Ignorable(u):
	# http://www.unicode.org/reports/tr44/#Default_Ignorable_Code_Point
	#
	# TODO Move me to unicodedata module and autogenerate.
	#
	# Unicode 14.0:
	# $ grep '; Default_Ignorable_Code_Point ' DerivedCoreProperties.txt | sed 's/;.*#/#/'
	# 00AD          # Cf       SOFT HYPHEN
	# 034F          # Mn       COMBINING GRAPHEME JOINER
	# 061C          # Cf       ARABIC LETTER MARK
	# 115F..1160    # Lo   [2] HANGUL CHOSEONG FILLER..HANGUL JUNGSEONG FILLER
	# 17B4..17B5    # Mn   [2] KHMER VOWEL INHERENT AQ..KHMER VOWEL INHERENT AA
	# 180B..180D    # Mn   [3] MONGOLIAN FREE VARIATION SELECTOR ONE..MONGOLIAN FREE VARIATION SELECTOR THREE
	# 180E          # Cf       MONGOLIAN VOWEL SEPARATOR
	# 180F          # Mn       MONGOLIAN FREE VARIATION SELECTOR FOUR
	# 200B..200F    # Cf   [5] ZERO WIDTH SPACE..RIGHT-TO-LEFT MARK
	# 202A..202E    # Cf   [5] LEFT-TO-RIGHT EMBEDDING..RIGHT-TO-LEFT OVERRIDE
	# 2060..2064    # Cf   [5] WORD JOINER..INVISIBLE PLUS
	# 2065          # Cn       <reserved-2065>
	# 2066..206F    # Cf  [10] LEFT-TO-RIGHT ISOLATE..NOMINAL DIGIT SHAPES
	# 3164          # Lo       HANGUL FILLER
	# FE00..FE0F    # Mn  [16] VARIATION SELECTOR-1..VARIATION SELECTOR-16
	# FEFF          # Cf       ZERO WIDTH NO-BREAK SPACE
	# FFA0          # Lo       HALFWIDTH HANGUL FILLER
	# FFF0..FFF8    # Cn   [9] <reserved-FFF0>..<reserved-FFF8>
	# 1BCA0..1BCA3  # Cf   [4] SHORTHAND FORMAT LETTER OVERLAP..SHORTHAND FORMAT UP STEP
	# 1D173..1D17A  # Cf   [8] MUSICAL SYMBOL BEGIN BEAM..MUSICAL SYMBOL END PHRASE
	# E0000         # Cn       <reserved-E0000>
	# E0001         # Cf       LANGUAGE TAG
	# E0002..E001F  # Cn  [30] <reserved-E0002>..<reserved-E001F>
	# E0020..E007F  # Cf  [96] TAG SPACE..CANCEL TAG
	# E0080..E00FF  # Cn [128] <reserved-E0080>..<reserved-E00FF>
	# E0100..E01EF  # Mn [240] VARIATION SELECTOR-17..VARIATION SELECTOR-256
	# E01F0..E0FFF  # Cn [3600] <reserved-E01F0>..<reserved-E0FFF>
	return (
		u == 0x00AD or				# Cf       SOFT HYPHEN
		u == 0x034F or				# Mn       COMBINING GRAPHEME JOINER
		u == 0x061C or				# Cf       ARABIC LETTER MARK
		0x115F <= u <= 0x1160 or	# Lo   [2] HANGUL CHOSEONG FILLER..HANGUL JUNGSEONG FILLER
		0x17B4 <= u <= 0x17B5 or	# Mn   [2] KHMER VOWEL INHERENT AQ..KHMER VOWEL INHERENT AA
		0x180B <= u <= 0x180D or	# Mn   [3] MONGOLIAN FREE VARIATION SELECTOR ONE..MONGOLIAN FREE VARIATION SELECTOR THREE
		u == 0x180E or				# Cf       MONGOLIAN VOWEL SEPARATOR
		u == 0x180F or				# Mn       MONGOLIAN FREE VARIATION SELECTOR FOUR
		0x200B <= u <= 0x200F or	# Cf   [5] ZERO WIDTH SPACE..RIGHT-TO-LEFT MARK
		0x202A <= u <= 0x202E or	# Cf   [5] LEFT-TO-RIGHT EMBEDDING..RIGHT-TO-LEFT OVERRIDE
		0x2060 <= u <= 0x2064 or	# Cf   [5] WORD JOINER..INVISIBLE PLUS
		u == 0x2065 or				# Cn       <reserved-2065>
		0x2066 <= u <= 0x206F or	# Cf  [10] LEFT-TO-RIGHT ISOLATE..NOMINAL DIGIT SHAPES
		u == 0x3164 or				# Lo       HANGUL FILLER
		0xFE00 <= u <= 0xFE0F or	# Mn  [16] VARIATION SELECTOR-1..VARIATION SELECTOR-16
		u == 0xFEFF or				# Cf       ZERO WIDTH NO-BREAK SPACE
		u == 0xFFA0 or				# Lo       HALFWIDTH HANGUL FILLER
		0xFFF0 <= u <= 0xFFF8 or	# Cn   [9] <reserved-FFF0>..<reserved-FFF8>
		0x1BCA0 <= u <= 0x1BCA3 or	# Cf   [4] SHORTHAND FORMAT LETTER OVERLAP..SHORTHAND FORMAT UP STEP
		0x1D173 <= u <= 0x1D17A or	# Cf   [8] MUSICAL SYMBOL BEGIN BEAM..MUSICAL SYMBOL END PHRASE
		u == 0xE0000 or				# Cn       <reserved-E0000>
		u == 0xE0001 or				# Cf       LANGUAGE TAG
		0xE0002 <= u <= 0xE001F or	# Cn  [30] <reserved-E0002>..<reserved-E001F>
		0xE0020 <= u <= 0xE007F or	# Cf  [96] TAG SPACE..CANCEL TAG
		0xE0080 <= u <= 0xE00FF or	# Cn [128] <reserved-E0080>..<reserved-E00FF>
		0xE0100 <= u <= 0xE01EF or	# Mn [240] VARIATION SELECTOR-17..VARIATION SELECTOR-256
		0xE01F0 <= u <= 0xE0FFF or	# Cn [3600] <reserved-E01F0>..<reserved-E0FFF>
		False)
