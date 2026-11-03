from .ials import IALS


class CustomALS(IALS):
    def target(self, r):
        return (r - 3) / 2
