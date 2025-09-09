from ChessBoardNN import ChessBoardNN
import unittest

class NetTest(unittest.TestCase):
    def setUp(self):
        self.net = ChessBoardNN()
        self.net.setPrintBoard()
        self.net._printToConsole = False

    def process(self, input):
        cyclesRet = self.net.processInput(input)
        ret = self.net.flushOutput()
        #print(ret)
        return ret

    def testTokenize(self):
        s = self.net.tokenize("""1 E2 E4,
2 g8 f6""")
        self.assertEqual(s, ['1', 'e2', 'e4', ',', '2', 'g8', 'f6'])
    
    def testPawn(self):
        s = self.process("e2 e4")
        self.assertEqual(s, """ |ABCDEFGH
8|rnbqkbnr
7|pppppppp
6|........
5|........
4|....P...
3|........
2|PPPP.PPP
1|RNBQKBNR
""")
        
    def testPawnEnPassant(self):
        s = self.process("""en passant:
1 E2 E4
2 g8 f6
3 e4 e5
4 d7 d5
5 e5 d6""")
        self.assertEqual(s, """ |ABCDEFGH
8|rnbqkbnr
7|pppppppp
6|........
5|........
4|....P...
3|........
2|PPPP.PPP
1|RNBQKBNR
 |ABCDEFGH
8|rnbqkb.r
7|pppppppp
6|.....n..
5|........
4|....P...
3|........
2|PPPP.PPP
1|RNBQKBNR
 |ABCDEFGH
8|rnbqkb.r
7|pppppppp
6|.....n..
5|....P...
4|........
3|........
2|PPPP.PPP
1|RNBQKBNR
 |ABCDEFGH
8|rnbqkb.r
7|ppp.pppp
6|.....n..
5|...pP...
4|........
3|........
2|PPPP.PPP
1|RNBQKBNR
 |ABCDEFGH
8|rnbqkb.r
7|ppp.pppp
6|...P.n..
5|........
4|........
3|........
2|PPPP.PPP
1|RNBQKBNR
""")
        
    def testCastlings(self):
        # prepare - move figures to allow kings to move
        self.net.setPrintBoard(False) 
        s = self.process("""castlings
a2 a4
b2 b4
c2 c4
d2 d4
e2 e4
f2 f4
g2 g4
h2 h4

a7 a5
b7 b5
c7 c5
d7 d5
e7 e5
f7 f5
g7 g5
h7 h5

b1 a3
c1 d2
d1 e2
f1 g2
g1 f3

b8 a6
c8 d7
d8 e7
f8 g7
""")
        self.net.setPrintBoard(True)
        beforeCastlings = self.process("g8 f6")
        self.assertEqual(beforeCastlings, """ |ABCDEFGH
8|r...k..r
7|...bq.b.
6|n....n..
5|pppppppp
4|PPPPPPPP
3|N....N..
2|...BQ.B.
1|R...K..R
""")
        # castlings for both kings:
        # another castlings are:
        # e1 g1, e8 g8
        s = self.process("e1 c1, e8 c8")
        self.assertEqual(s, """ |ABCDEFGH
8|r...k..r
7|...bq.b.
6|n....n..
5|pppppppp
4|PPPPPPPP
3|N....N..
2|...BQ.B.
1|..KR...R
 |ABCDEFGH
8|..kr...r
7|...bq.b.
6|n....n..
5|pppppppp
4|PPPPPPPP
3|N....N..
2|...BQ.B.
1|..KR...R
""")

if __name__ == "__main__":
    # launch: python -m unittest -v
    unittest.main()
    #exit(0)
    #n = NetTest()
    #n.setUp()
    #n.testPawnEnPassant()