

class UFOLibError(Exception):
    pass


class UnsupportedUFOFormat(UFOLibError):
    pass


class GlifLibError(UFOLibError):
    pass


class UnsupportedGLIFFormat(GlifLibError):
    pass
