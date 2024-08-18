def Sum(L):
    count =0
    for x in L:
        count+=x
    return count


def SumRecursiva(L, i, n, count):

    
    # Base case
    if n <= i:
        return count
    
    count += L[i]
    
    # Going into the recursion
    count = SumRecursiva(L, i + 1, n, count)
    
    return count


L = [1, 2, 3, 4, 5]

print(Sum(L))

# Driver's code

count = 0
n = len(L)
print(SumRecursiva(L, 0, n, count))

