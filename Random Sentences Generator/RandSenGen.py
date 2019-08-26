class Random:
    def __init__(self, seed):
        self.random = seed

    def next(self):
        self.random = ((pow(7, 5) * self.random) % (pow(2, 31) - 1))
        return self.random

    def choose(self, objects):
        self.obj = objects
        fin = self.obj[self.next()%len(self.obj)]
        return fin

class Grammar:
    def __init__(self, seed):
        self.random = Random(seed)
        self.dict = {}

    def rule(self, left, right):
        self.left = left
        self.right = right
        if self.left not in self.dict:
            self.dict[left] = [right]
        else:
            self.dict[left] += [right]


    def generate(self):
        if 'Start' in self.dict:
            tmp = self.generating(('Start',))
            return tmp
        else:
            raise RuntimeError("Can't start without 'Start' command")


    def generating(self, strings):
        stringy = ''
        for x in strings:
            if x not in self.dict:
                stringy += x + ' '
            else:
                recur = self.random.choose(self.dict[x])
                stringy += self.generating(recur)
        return stringy






# R = Random(101)
# print(R.next())
# print(R.next())
# print(R.choose((1,2,3,4)))
# print(R.choose((1,2,3,4)))
# print(R.choose((1,2,3,4)))
# print(R.choose((1,2,3,4,5,6)))
# print(R.choose((1,2,3,4,5,6,7,8)))
# print(R.choose((7,3,5,2,7)))
# print(R.choose((6,2,1,7,4)))
# print(R.choose((0,0,0,0,0)))     # Always 0
# print(R.choose((1,1,1,1)))       # Always 1
#
#
# G = Grammar(101)
# G.rule('Noun',   ('cat',))                                #  01
# print(G.dict)
# G.rule('Noun',   ('boy',))                                #  02
# print(G.dict)
# G.rule('Noun',   ('dog',))                                #  03
# print(G.dict)
# G.rule('Noun',   ('girl',))                               #  04
# print(G.dict)
# G.rule('Verb',   ('bit',))                                #  05
# print(G.dict)
# G.rule('Verb',   ('chased',))                             #  06
# print(G.dict)
# G.rule('Verb',   ('kissed',))                             #  07
# print(G.dict)
# G.rule('Phrase', ('the', 'Noun', 'Verb', 'the', 'Noun'))  #  08
# print(G.dict)
# G.rule('Story',  ('Phrase',))                             #  09
# print(G.dict)
# G.rule('Story',  ('Phrase', 'and', 'Story'))              #  10
# print(G.dict)
# G.rule('Start',  ('Story', '.'))                          #  11
# print(G.dict)
#
# print(G.generate())
# print(G.generate())
# print(G.generate())
# print(G.generate())      # The first one are the same as the ones from the assignment
# print(G.generate())
# print(G.generate())
# print(G.generate())
# print(G.generate())
# print(G.generate())
# print(G.generate())
# print(G.generate())