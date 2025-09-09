*ChessBoard Neural Network*
Converts chessboard moves, like `e2 e4`, into actual chessboard representation:
'''
> test; e2 and then e4
 |ABCDEFGH
8|rnbqkbnr
7|pppppppp
6|........
5|........
4|....P...
3|........
2|PPPP.PPP
1|RNBQKBNR

> a7 a6
 |ABCDEFGH
8|rnbqkbnr
7|.ppppppp
6|p.......
5|........
4|....P...
3|........
2|PPPP.PPP
1|RNBQKBNR
'''

Purpose of this network is to embed it into actual neural engine as an inpul layer.
Main output is not console text, but rather an embedding:
```
> repr
Current chessboard representation by the network in memory:
 RNBQKPrnbqkp.|RNBQKPrnbqkp.|RNBQKPrnbqkp.|RNBQKPrnbqkp.|RNBQKPrnbqkp.|RNBQKPrnbqkp.|RNBQKPrnbqkp.|RNBQKPrnbqkp.|
  A           | B           | C           | D           | E           | F           | G           | H           |
[1............|.1...........|..1..........|...1.........|....1........|..1..........|.1...........|1............
 .....1.......|.....1.......|.....1.......|.....1.......|............1|.....1.......|.....1.......|.....1.......
 ............1|............1|............1|............1|............1|............1|............1|............1
 ............1|............1|............1|............1|.....1.......|............1|............1|............1
 ............1|............1|............1|............1|............1|............1|............1|............1
 ...........1.|............1|............1|............1|............1|............1|............1|............1
 ............1|...........1.|...........1.|...........1.|...........1.|...........1.|...........1.|...........1.
 ......1......|.......1.....|........1....|.........1...|..........1..|........1....|.......1.....|......1......]
```

*Class diagram*


*Modes*
Newly created network doesn't print anything to console for faster operation. This is controled by activation on neuron [0].
```
    def setPrintBoard(self, usePrint: bool = True):
        # This is special neuron that tweaks net to print board in ascii mode after each move.
        self._activations[0] = 1 if usePrint else 0
        # Asii printing is slow - few cycles for each letter.
        # If you use network to embed into another,
        # use embedding() to obtain inner state faster and directly
```
All further printing work is done using standard feed forward cycle.

* Network size*
```
> info
Neurons: 5553
Links: 18287
Active neurons: 256
Average connectivity: 3.29
```
If to exclude neurons which make it possible for ASCII printing, network size will decrease to 4521 neurons (-1032 neurons, 22%) and 15455 links (-2832 links, 18%).
Unzipped file ChessBoardNN.py is 153Kb big, zipped - 28 Kb.

*Development*
Network was created in specialized IDE, then exported to Python.