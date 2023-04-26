outputfile_folder = './correctoutput'

charcount = 0
charmatch = 0
answer = []

# open the answer
with open(outputfile_folder+'/c_1.txt', 'r') as file:
    while True:
        char = file.read(1)
        if not char:
            # 文件末尾
            print(charcount)
            break
        else :
            answer.append(char)
            charcount += 1

# open file to check
with open(outputfile_folder+'/c_1.txt', 'r') as file:
    charmatch = 0
    for char_i in range(charcount):
        char = file.read(1)
        print(char_i)
        print(answer[char_i])

        if answer[char_i] == char:
            charmatch += 1
    print(charmatch)
