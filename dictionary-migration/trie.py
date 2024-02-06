class Trie:
    def __init__(self):
        self._value = None
        self._map = {}

    def store(self, path, value, cover=True):
        if len(path) == 0:
            if cover or self._value is None:
                self._value = value
        else:
            if path[0] not in self._map:
                self._map[path[0]] = Trie()
            self._map[path[0]].store(path[1:], value, cover=cover)

    def forward(self, token, ignore=False):
        if token in self._map:
            return self._map[token]
        elif ignore:
            return None
        else:
            raise KeyError(f'token "{token}" not found')

    def value(self):
        return self._value

    def search(self, path):
        if len(path) == 0:
            if self._value is None:
                raise KeyError('root is not a terminator')
            return self._value
        current = self
        for token in path:
            if current is None:
                raise KeyError(f'path "{path}" not found')
            current = current.forward(token, ignore=True)
        if current._value is None:
            raise KeyError(f'path "{path}" is not a terminator')
        return current._value
