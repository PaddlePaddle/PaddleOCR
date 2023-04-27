from jiwer import wer

correctfile_folder = './correctoutput'
comparefile_folder = './outputfile'

reference = ''
hypothesis = ''




# open the answer
with open(correctfile_folder+'/c_1.txt', 'r', encoding="utf-8") as file:
    reference = ''
    while True:
        line = file.readline()
        if not line:
            break
        line_into_list = line.strip()
        reference+=line_into_list
print(reference)
        
print("--------------------------------------------------")


# open file to check
with open(comparefile_folder+'/1.jpg.txt', 'r', encoding="utf-8") as file:
    while True:
        line = file.readline()
        if not line:
            break
        line_into_list = line.strip()
        hypothesis+=line_into_list
print(hypothesis)

wer_rate = wer(reference, hypothesis)

print('WER=%2f'%wer_rate)