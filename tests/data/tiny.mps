NAME          tiny
ROWS
 N  COST
 E  r1
 G  r2
 L  r3
 L  cap
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    x1        COST                 2   r1                   1
    x1        r2                   1   cap                  2
    x2        COST                 3   r1                   1
    x2        r3                   1   cap                  3
    x3        COST                 4   r2                   1
    x3        r3                   1
    MARK0001  'MARKER'                 'INTEND'
RHS
    rhs       r1                   1   r2                   1
    rhs       r3                   1   cap                  4
    rhs       COST                -5
BOUNDS
 UP bnd       x1                   1
 UP bnd       x2                   1
 BV bnd       x3
ENDATA
