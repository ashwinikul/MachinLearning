# File Read program

#!/usr/bin/python 

sum = 0 #storing Sum Value
l=1 #storing Line No.
# Open a file, by default read mode
fo = open("Input.txt")

for line in fo:
        sum = 0
        
        for num in line.strip().split(','):
            sum += int(num)
        print 'Line :',l,'Contains:' ,line.rstrip('\n')
        print 'Sum of Numbers in Line :',l,' is',sum,'\n\n'
        l +=1
        
# Close opend file
fo.close()
