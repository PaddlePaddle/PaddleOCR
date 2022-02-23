import matplotlib.type1font as t1f
import os.path
import difflib


def test_Type1Font():
    filename = os.path.join(os.path.dirname(__file__), 'cmr10.pfb')
    font = t1f.Type1Font(filename)
    slanted = font.transform({'slant': 1})
    condensed = font.transform({'extend': 0.5})
    with open(filename, 'rb') as fd:
        rawdata = fd.read()
    assert font.parts[0] == rawdata[0x0006:0x10c5]
    assert font.parts[1] == rawdata[0x10cb:0x897f]
    assert font.parts[2] == rawdata[0x8985:0x8ba6]
    assert font.parts[1:] == slanted.parts[1:]
    assert font.parts[1:] == condensed.parts[1:]
    assert font.decrypted.startswith(b'dup\n/Private 18 dict dup begin')
    assert font.decrypted.endswith(b'mark currentfile closefile\n')

    differ = difflib.Differ()
    diff = list(differ.compare(
        font.parts[0].decode('latin-1').splitlines(),
        slanted.parts[0].decode('latin-1').splitlines()))
    for line in (
         # Removes UniqueID
         '- FontDirectory/CMR10 known{/CMR10 findfont dup/UniqueID known{dup',
         '+ FontDirectory/CMR10 known{/CMR10 findfont dup',
         # Changes the font name
         '- /FontName /CMR10 def',
         '+ /FontName /CMR10_Slant_1000 def',
         # Alters FontMatrix
         '- /FontMatrix [0.001 0 0 0.001 0 0 ]readonly def',
         '+ /FontMatrix [0.001 0 0.001 0.001 0 0]readonly def',
         # Alters ItalicAngle
         '-  /ItalicAngle 0 def',
         '+  /ItalicAngle -45.0 def'):
        assert line in diff, 'diff to slanted font must contain %s' % line

    diff = list(differ.compare(
        font.parts[0].decode('latin-1').splitlines(),
        condensed.parts[0].decode('latin-1').splitlines()))
    for line in (
         # Removes UniqueID
         '- FontDirectory/CMR10 known{/CMR10 findfont dup/UniqueID known{dup',
         '+ FontDirectory/CMR10 known{/CMR10 findfont dup',
         # Changes the font name
         '- /FontName /CMR10 def',
        '+ /FontName /CMR10_Extend_500 def',
         # Alters FontMatrix
         '- /FontMatrix [0.001 0 0 0.001 0 0 ]readonly def',
         '+ /FontMatrix [0.0005 0 0 0.001 0 0]readonly def'):
        assert line in diff, 'diff to condensed font must contain %s' % line


def test_overprecision():
    # We used to output too many digits in FontMatrix entries and
    # ItalicAngle, which could make Type-1 parsers unhappy.
    filename = os.path.join(os.path.dirname(__file__), 'cmr10.pfb')
    font = t1f.Type1Font(filename)
    slanted = font.transform({'slant': .167})
    lines = slanted.parts[0].decode('ascii').splitlines()
    matrix, = [line[line.index('[')+1:line.index(']')]
               for line in lines if '/FontMatrix' in line]
    angle, = [word
              for line in lines if '/ItalicAngle' in line
              for word in line.split() if word[0] in '-0123456789']
    # the following used to include 0.00016700000000000002
    assert matrix == '0.001 0 0.000167 0.001 0 0'
    # and here we had -9.48090361795083
    assert angle == '-9.4809'


def test_encrypt_decrypt_roundtrip():
    data = b'this is my plaintext \0\1\2\3'
    encrypted = t1f.Type1Font._encrypt(data, 'eexec')
    decrypted = t1f.Type1Font._decrypt(encrypted, 'eexec')
    assert encrypted != decrypted
    assert data == decrypted
