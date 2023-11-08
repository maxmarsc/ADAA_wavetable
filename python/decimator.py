import numpy as np


class Decimator9:
    def __init__(self):
        self.__h0: float = 8192 / 16384.0
        self.__h1: float = 5042 / 16384.0
        self.__h3: float = -1277 / 16384.0
        self.__h5: float = 429 / 16384.0
        self.__h7: float = -116 / 16384.0
        self.__h9: float = 18 / 16384.0
        self.__r1: float = 0.0
        self.__r2: float = 0.0
        self.__r3: float = 0.0
        self.__r4: float = 0.0
        self.__r5: float = 0.0
        self.__r6: float = 0.0
        self.__r7: float = 0.0
        self.__r8: float = 0.0
        self.__r9: float = 0.0

    def process(self, x0: float, x1: float) -> float:
        h9x0 = self.__h9 * x0
        h7x0 = self.__h7 * x0
        h5x0 = self.__h5 * x0
        h3x0 = self.__h3 * x0
        h1x0 = self.__h1 * x0
        self.__r10 = self.__r9 + h9x0
        self.__r9 = self.__r8 + h7x0
        self.__r8 = self.__r7 + h5x0
        self.__r7 = self.__r6 + h3x0
        self.__r6 = self.__r5 + h1x0
        self.__r5 = self.__r4 + h1x0 + self.__h0 * x1
        self.__r4 = self.__r3 + h3x0
        self.__r3 = self.__r2 + h5x0
        self.__r2 = self.__r1 + h7x0
        self.__r1 = h9x0
        return self.__r10


class Decimator17:
    def __init__(self):
        self.__h0: float = 0.5
        self.__h1: float = 0.314356238
        self.__h3: float = -0.0947515890
        self.__h5: float = 0.0463142134
        self.__h7: float = -0.0240881704
        self.__h9: float = 0.0120250406
        self.__h11: float = -0.00543170841
        self.__h13: float = 0.00207426259
        self.__h15: float = -0.000572688237
        self.__h17: float = 5.18944944e-005
        self.__r1: float = 0.0
        self.__r2: float = 0.0
        self.__r3: float = 0.0
        self.__r4: float = 0.0
        self.__r5: float = 0.0
        self.__r6: float = 0.0
        self.__r7: float = 0.0
        self.__r8: float = 0.0
        self.__r9: float = 0.0
        self.__r10: float = 0.0
        self.__r11: float = 0.0
        self.__r12: float = 0.0
        self.__r13: float = 0.0
        self.__r14: float = 0.0
        self.__r15: float = 0.0
        self.__r16: float = 0.0
        self.__r17: float = 0.0

    def process(self, x0: float, x1: float) -> float:
        h17x0 = self.__h17 * x0
        h15x0 = self.__h15 * x0
        h13x0 = self.__h13 * x0
        h11x0 = self.__h11 * x0
        h9x0 = self.__h9 * x0
        h7x0 = self.__h7 * x0
        h5x0 = self.__h5 * x0
        h3x0 = self.__h3 * x0
        h1x0 = self.__h1 * x0
        self.__r18 = self.__r17 + h17x0
        self.__r17 = self.__r16 + h15x0
        self.__r16 = self.__r15 + h13x0
        self.__r15 = self.__r14 + h11x0
        self.__r14 = self.__r13 + h9x0
        self.__r13 = self.__r12 + h7x0
        self.__r12 = self.__r11 + h5x0
        self.__r11 = self.__r10 + h3x0
        self.__r10 = self.__r9 + h1x0
        self.__r9 = self.__r8 + h1x0 + self.__h0 * x1
        self.__r8 = self.__r7 + h3x0
        self.__r7 = self.__r6 + h5x0
        self.__r6 = self.__r5 + h7x0
        self.__r5 = self.__r4 + h9x0
        self.__r4 = self.__r3 + h11x0
        self.__r3 = self.__r2 + h13x0
        self.__r2 = self.__r1 + h15x0
        self.__r1 = h17x0
        return self.__r18
