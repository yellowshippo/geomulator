

class DifferentialLabel(str):

    def __new__(cls, differential_orders):
        obj = str.__new__(cls, '-'.join(str(o) for o in differential_orders))
        return obj

    def to_list(self):
        return [int(s) for s in self.split('-')]

    def add_count(self, index):
        list_dl = self.to_list()
        list_dl[index] += 1
        return DifferentialLabel(list_dl)


# dl = DifferentialLabel([3, 4, 1, 0])
# print(dl)
# print(dl.increase(3))
