#strings are stored as arrays
word = 'TEST Test TEST'

#substring
word[1:4]
word[:4]
word[5:]

#same as find in Excel, returns the position of the string
word.index(' ', 2)

%matplotlib

for i in range(2):
    b=_.join(str(i))