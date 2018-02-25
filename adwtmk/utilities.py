def get_all_bits(byte_string: bytes, reverse: bool=False)->list:
    if not reverse:
        return [bit
                for L in map(lambda byte: [((byte & (1 << i)) >> i) for i in range(7, -1, -1)], byte_string)
                for bit in L]
    else:
        return [bit
                for L in map(lambda byte: [((byte & (1 << i)) >> i) for i in range(8)], byte_string)
                for bit in L]
