from math import sqrt


def test():
    print("hello world")


test()


def giai_ptb2(a, b, c):
    deta = b * b - 4 * a * c
    if deta < 0:
        print("pt vo nghiem")
    elif deta == 0:
        print("pt co 1 nghiem")
        x = -b / (2 * a)
        print(f"nghiem cua pt: x = {x}")
    else:
        print("pt co 2 nghiem")
        x1 = (-b + sqrt(deta)) / (2 * a)
        x2 = (-b - sqrt(deta)) / (2 * a)
        print(f"nghiem cua pt: x1 = {x1}, x2 = {x2}")

giai_ptb2(4, -5, 1)
