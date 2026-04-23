def test_struct():
    list_item = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    print(f" item 0 = {list_item[0]}")

    tl = (0,1,2,3,4,5,6,7,8,9)
    print(f" tl = {tl[2]}")

class Car:
    def __init__(self, name, color):
        self.name = name
        self.color = color

    def drive(self):
        print(f" {self.name} {self.color}")

test_struct()
my_car = Car("BMW", "red")
my_car.drive()