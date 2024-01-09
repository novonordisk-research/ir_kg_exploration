def remove_prefix(s, prefix):
    if s.startswith(prefix):
        s = s[len(prefix) :]
    return s
