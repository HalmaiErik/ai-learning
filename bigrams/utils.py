def build_char_index_maps(words: list[str]):
    chars = sorted(list(set(''.join(words))))

    stoi = {s:i+1 for i, s in enumerate(chars)}
    stoi['.'] = 0

    itos = {i:s for s, i in stoi.items()}

    return stoi, itos