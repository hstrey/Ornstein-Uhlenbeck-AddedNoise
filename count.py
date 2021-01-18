count = 0
for i in range(1,16):
    for j in range(1,16):
        if 1.0*i*j<16.0:
            count = count+1
            print(i,j)
print(count)
