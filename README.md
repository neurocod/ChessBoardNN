ChessBoard Neural Network
=========================

A neural network that converts chess moves (e.g. `e2 e4`) into an actual **chessboard representation**.  
It can be used as an **input layer** for a larger neural engine.

* * *

Example
-------

```text
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
```

* * *

Embedding Output
----------------

The main output is not console text, but an **embedding** that represents the board in memory:

```text
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

* * *

Modes
-----

By default, the network does not print anything to the console for faster operation.  
This behavior is controlled by **neuron \[0\]**:

```python
def setPrintBoard(self, usePrint: bool = True):
    # This special neuron toggles ASCII board printing after each move.
    self._activations[0] = 1 if usePrint else 0
    # ASCII printing is slow — several cycles per character.
    # For embedding into another system, call embedding() to obtain
    # the internal state faster and directly.
```

Printing is handled via the standard feed-forward cycle.

* * *

Network Info
------------

```text
> info
Neurons: 5553
Links: 18287
Active neurons: 256
Average connectivity: 3.29
```

If ASCII-printing neurons are excluded:

*   **4521 neurons** (–1032, ~22%)
*   **15455 links** (–2832, ~18%)

File sizes:

*   **ChessBoardNN.py** — 153 KB (unzipped)
*   **ChessBoardNN.py.zip** — 28 KB

* * *

Development
-----------

*   The network was created in a **specialized IDE**.
*   Exported to **Python** for integration into other projects.

* * *

Class Diagram
-------------

_(to be added)_