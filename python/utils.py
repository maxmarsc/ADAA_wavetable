def table_size(n, init_size):
    sm = 0
    for i in range(n):
        sm += init_size / (2**i)
    return sm

# print(table_size(8, 2048))