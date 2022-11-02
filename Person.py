class Human:
    # class variable
    name = None
    age = 0
    # method
    def __init__(self, name, age): # constructor
        self.name = name
        self.age = age
    def hello(self):
        print('hello')
    def getOld(self):
        self.age += 1

if __name__ == "__main__":
    p1 = Human('kim', 22)
    p2 = Human('lee', 26)
    p2.getOld()
    print(p1.name, p1.age)
    print(p2.name, p2.age)