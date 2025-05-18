def rec(a):
    if a <= 0:
        return 1
    return 1 + rec(a-2)

def rec(a, b):
    if a <=0 or b <= 0:
        print (rec(a)+rec(b))
        rec(a-b, b-a)

rec(4, 2)
